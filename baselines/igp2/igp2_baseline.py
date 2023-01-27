import copy
import os
import pandas as pd
import pickle
import dill
import concurrent.futures
import argparse
import sys
import time

from igp2 import setup_logging, AgentState
from igp2.data.data_loaders import InDDataLoader
from igp2.goal import PointGoal
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString
from igp2.recognition.goalrecognition import *
from igp2.recognition.astar import AStar
from igp2.cost import Cost
from igp2.results import *
from igp2.planlibrary.maneuver import Maneuver, SwitchLane
from igp2.planlibrary.macro_action import ChangeLane

from ogrit.core.base import set_working_dir, get_occlusions_dir, get_data_dir, get_igp2_results_dir
from ogrit.occlusion_detection.visualisation_tools import get_box

# Code adapted from https://github.com/uoe-agents/IGP2/blob/main/scripts/experiments/experiment_multi_process.py
def create_args():
    config_specification = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    config_specification.add_argument('--num_workers', default="0",
                                      help="Number of parralel processes. Set 0 for auto", type=int)
    config_specification.add_argument('--output', default="_igp2_baseline",
                                      help="Output .pkl filename", type=str)
    config_specification.add_argument('--tuning', default="0",
                                      help="0: runs default tuning parameters defined in the scenario JSON. \
                                      1: runs all cost parameters defined in the script.", type=int)
    config_specification.add_argument('--reward_scheme', default="1",
                                      help="0: runs the default reward scheme defined in the IGP2 paper. \
                                      1: runs alternate reward scheme defined in Cost.cost_difference_resampled().",
                                      type=int)
    config_specification.add_argument('--dataset', default="test",
                                      help="valid: runs on the validation dataset, test: runs on the test dataset",
                                      type=str)
    config_specification.add_argument('--scenario', default="heckstrasse", type=str)

    parsed_config_specification = vars(config_specification.parse_args())
    return parsed_config_specification


def extract_goal_data(goals_data):
    """Creates a list of Goal objects from the .json scenario file data"""
    goals = []
    for goal_data in goals_data:
        point = Point(np.array(goal_data))
        goals.append(PointGoal(point, 2.))

    return goals


def read_and_process_data(scenario, episode_id):
    """Identifies which frames and agents to perform goal recognition for in episode,
    from a provided .csv file."""

    file_path = get_data_dir() + f'/{scenario}_e{episode_id}.csv'
    data = pd.read_csv(file_path).drop_duplicates(['frame_id', 'agent_id', 'ego_agent_id'])
    return data


def goal_recognition_agent(frames, recordingID, framerate, aid, data, goal_recognition: GoalRecognition,
                           goal_probabilities: GoalsProbabilities):
    """
    Computes the goal recognition results for specified agent at specified frames.
    Note: the frames are already adjusted for occlusions (i.e., only those in which the ego saw the target and
          where we used A* to fill in the occluded frames).
    """

    goal_probabilities_c = copy.deepcopy(goal_probabilities)
    result_agent = None
    for frame in frames:
        if aid in frame.dead_ids:
            frame.dead_ids.remove(aid)

    initial_frame = min(data['frame_id'])

    # For each sample in the OGRIT dataset, collect the goal probabilities.
    for _, row in data.iterrows():
        try:
            if result_agent is None:
                result_agent = AgentResult(row['true_goal'])
            frame_id = int(row['frame_id'])

            # Use the states of the vehicle we have seen so far to predict its goal.
            agent_states = [frame.agents[aid] for frame in frames[0:frame_id - initial_frame + 1]]
            trajectory = ip.StateTrajectory(framerate, frames=agent_states)

            current_frame = frames[frame_id - initial_frame].agents

            t_start = time.perf_counter()
            goal_probabilities_c = goal_recognition.update_goals_probabilities(goal_probabilities_c, trajectory, aid,
                                                                               frame_ini=frames[0].agents,
                                                                               frame=current_frame, maneuver=None)
            t_end = time.perf_counter()

            result_agent.add_data((frame_id, copy.deepcopy(goal_probabilities_c), t_end - t_start, trajectory.path[-1],
                                   row['fraction_observed']))

        except Exception as e:
            logger.error(f"Fatal in recording_id: {recordingID} for aid: {aid} at frame {frame_id}.")
            logger.error(f"Error message: {str(e)}")

    return aid, result_agent


def multi_proc_helper(arg_list):
    """Allows to pass multiple arguments as multiprocessing routine."""
    return goal_recognition_agent(arg_list[0], arg_list[1], arg_list[2], arg_list[3], arg_list[4], arg_list[5],
                                  arg_list[6])


def run_experiment(cost_factors: Dict[str, float] = None, use_priors: bool = False, max_workers: int = None):
    """Run goal prediction in parralel for each agent, across all specified scenarios."""
    result_experiment = ExperimentResult()

    for scenario_name in SCENARIOS:
        scenario_map = ip.Map.parse_from_opendrive(f"scenarios/maps/{scenario_name}.xodr")

        data_loader = InDDataLoader(f"scenarios/configs/{scenario_name}.json", [DATASET])
        data_loader.load()

        # Scenario specific parameters
        SwitchLane.TARGET_SWITCH_LENGTH = data_loader.scenario.config.target_switch_length
        ChangeLane.CHECK_ONCOMING = data_loader.scenario.config.check_oncoming
        ip.Trajectory.VELOCITY_STOP = 1.
        Maneuver.NORM_WIDTH_ACCEPTABLE = 0.5
        Maneuver.LON_SWERVE_DISTANCE = 8

        if cost_factors is None:
            cost_factors = data_loader.scenario.config.cost_factors
        episode_ids = data_loader.scenario.config.dataset_split[DATASET]
        test_data = [read_and_process_data(scenario_name, episode_id) for episode_id in episode_ids]

        goals_data = data_loader.scenario.config.goals
        if use_priors:
            goals_priors = data_loader.scenario.config.goals_priors
        else:
            goals_priors = None

        goals = extract_goal_data(goals_data)
        goal_probabilities = GoalsProbabilities(goals, priors=goals_priors)
        astar = AStar(next_lane_offset=0.25)
        cost = Cost(factors=cost_factors)
        ind_episode = 0

        with open(get_occlusions_dir() + f"{scenario_name}_e{episode_ids[ind_episode]}.p", 'rb') as file:
                occlusions = pickle.load(file)

        for episode in data_loader:
            # episode specific parameters
            Maneuver.MAX_SPEED = episode.metadata.max_speed  # Can be set explicitly if the episode provides a speed limit
            cost.limits["velocity"] = episode.metadata.max_speed

            recordingID = episode.metadata.config['recordingId']
            framerate = episode.metadata.frame_rate
            logger.info(
                f"Starting experiment in scenario: {scenario_name}, episode_id: {episode_ids[ind_episode]},"
                f" recording_id: {recordingID}")
            smoother = ip.VelocitySmoother(vmin_m_s=ip.Trajectory.VELOCITY_STOP, vmax_m_s=episode.metadata.max_speed,
                                           n=10, amax_m_s2=5, lambda_acc=10)
            goal_recognition = GoalRecognition(astar=astar, smoother=smoother, scenario_map=scenario_map, cost=cost,
                                               reward_as_difference=REWARD_AS_DIFFERENCE)
            result_episode = EpisodeResult(episode.metadata, episode_ids[ind_episode], cost_factors)

            # Prepare inputs for multiprocessing
            grouped_data = test_data[ind_episode].groupby(['agent_id', 'ego_agent_id'])
            args = []
            for (aid, ego_id), group in grouped_data:
                data = group.copy()

                frame_ini = data.frame_id.values[0]
                frame_last = data.frame_id.values[-1]
                frames = episode.frames[frame_ini:frame_last + 1]
                frames = _get_occlusion_frames(frames, framerate, occlusions, aid, ego_id, goal_recognition,
                                               scenario_map)
                arg = [frames, recordingID, framerate, aid, data, goal_recognition, goal_probabilities]
                args.append(copy.deepcopy(arg))

            # Perform multiprocessing
            results_agents = []

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # with MockProcessPoolExecutor() as executor:
                results_agents = [executor.submit(multi_proc_helper, arg) for arg in args]
                for result_agent in concurrent.futures.as_completed(results_agents):
                    try:
                        result_episode.add_data(result_agent.result())
                    except Exception as e:
                        logger.error(f"Error during multiprocressing. Error message: {str(e)}")

            result_experiment.add_data((episode.metadata.config['recordingId'], copy.deepcopy(result_episode)))
            ind_episode += 1

    return result_experiment


def _generate_occluded_trajectory(goal_recognition, aid, last_seen_frame, framerate, frame_visible_again, scenario_map):
    """
    Use A* to generate a trajectory between the last seen position and the currently visible position.
    Move the start and end position to the lane's midline as to ge
    Args:
        goal_recognition:    GoalRecognition object that keeps track of the probabilities assigned to each goal
        aid:                 id of the vehicle for which we are predicting the occluded frames
        last_seen_frame:     frame was last visible to the ego
        framerate:           framerate of the simulation
        frame_visible_again: frame visible to the ego after being occluded
        scenario_map:        map of the scenario of the simulation
    """
    # Generate trajectory for those frames in which the target was occluded.
    # Project the initial and final point to the midline.
    last_state = last_seen_frame.agents[aid]
    last_pos = [last_state.position[0], last_state.position[1]]
    old_lane = scenario_map.best_lane_at(last_pos, heading=last_state.heading, max_distance=0.2)

    state_trajectory = ip.StateTrajectory(framerate, [last_state])

    new_state = frame_visible_again.agents[aid]
    new_lane = scenario_map.best_lane_at(new_state.position, heading=new_state.heading, max_distance=0.2)

    # Place the last seen and new seen positions to be on the midline to facilitate IGP2 finding a path.
    if old_lane is not None:
        x, y = (old_lane.midline.interpolate(old_lane.midline.project(Point(last_state.position)))).xy
        last_state.position = [list(x)[0], list(y)[0]]

    if new_lane is not None:
        new_point_om = (new_lane.midline.interpolate(new_lane.midline.project(Point(new_state.position))))
        new_position = [new_point_om.x, new_point_om.y]
    else:
        new_point_om = new_state.position
        new_position = new_state.position

    last_seen_frame = last_seen_frame.agents
    last_seen_frame[aid] = last_state

    try:
        trajectory, _ = goal_recognition.generate_trajectory(n_trajectories=1,
                                                             agent_id=aid,
                                                             frame=last_seen_frame,
                                                             goal=PointGoal(new_point_om, 2),
                                                             state_trajectory=state_trajectory)

    except RuntimeError as e:
        logger.debug(e)

        known_path = np.array([last_state.position, new_position])

        last_velocity = np.sqrt(last_state.velocity[0]**2 + last_state.velocity[1]**2)
        new_velocity = np.sqrt(new_state.velocity[0]**2 + new_state.velocity[1]**2)

        known_velocities = np.array([last_velocity, new_velocity])
        trajectory = [ip.VelocityTrajectory(known_path, known_velocities)]

    # The trajectory will start and end on the midline, thus interpolate from the actual position to the actual goal
    return trajectory


def _get_occlusion_frames(frames, framerate, occlusions, aid, ego_id, goal_recognition, scenario_map):
    """If the target is occluded w.r.t the ego, use A* to find a trajectory between the last position in which the
    ego saw the target and the next frame in which the target is visible. Then, update the frames in this interval to
    use the predicted, and not the actual, trajectory.

    Args:
        frames:           all the real frames in the trajectory
        framerate:        framerate of the simulation
        occlusions:       dictionary with the occlusions indexed as {frame_id: {ego_id: {road_id: {lane_id: occlusions}}}
        aid:              id of the vehicle for which we are predicting the occluded frames
        ego_id:           id of the vehicle trying to predict the aid's goal
        goal_recognition: GoalRecognition object that keeps track of the probabilities assigned to each goal
        scenario_map:     map of the scenario of the simulation
    """

    idx_last_seen = 0
    occlusion_frames = []
    initial_frame_id = frames[0].time
    metadata = frames[0].agents[aid].metadata

    # Find the frames in which the aid is occluded w.r.t the ego vehicle and interpolate the trajectory in those cases.
    for i, frame in enumerate(frames):
        frame_id = frame.time
        ego_occlusions = occlusions[frame_id][ego_id]["occlusions"]
        target_occluded = LineString(get_box(frame.agents[aid]).boundary).buffer(0.001).within(ego_occlusions)

        # i-idx_last_seen is 1 if there are no occlusions since the last frame
        nr_occluded_frames = i-idx_last_seen-1

        if target_occluded:
            continue

        # If the target was visible in the previous frame, do nothing. Else, if it wasn't visible before, but now it is,
        # use A* to fill in the occluded path.
        if nr_occluded_frames > 0 and idx_last_seen != 0:

            # The trajectory will have length nr_occluded_frames+1 since the first frame will be the last th ego saw.
            trajectory = _generate_occluded_trajectory(goal_recognition, aid, frames[idx_last_seen], framerate, frame,
                                                       scenario_map)

            trajectory = trajectory[0]

             # todo: check what happens to the trajectory when there is only one step missing not 1 step if only one missing nr_occluded_frames == 1:

            # Case in which the target vehicle doesn't move while being occluded w.r.t the ego.
            if sum(trajectory.timesteps) == 0:
                new_path = [list(trajectory.path[0])] * nr_occluded_frames
                new_vel = [0] * nr_occluded_frames
                new_acc = [0] * nr_occluded_frames
                new_heading = [frame.agents[aid].heading] * nr_occluded_frames
            else:
                initial_heading = trajectory.heading[0]
                final_heading = trajectory.heading[-1]
                initial_direction = np.array([np.cos(initial_heading), np.sin(initial_heading)])
                final_direction = np.array([np.cos(final_heading), np.sin(final_heading)])
                cs_path = CubicSpline(trajectory.times, trajectory.path, bc_type=((1, initial_direction),
                                                                                  (1, final_direction)))

                cs_velocity = CubicSpline(trajectory.times, trajectory.velocity)
                cs_acceleration = CubicSpline(trajectory.times, trajectory.acceleration)
                cs_heading = CubicSpline(trajectory.times, trajectory.heading)

                ts = np.linspace(0, trajectory.times[-1], nr_occluded_frames)

                new_path = cs_path(ts)
                new_vel = cs_velocity(ts)
                new_acc = cs_acceleration(ts)
                new_heading = cs_heading(ts)

            # j starts at 1 since the first step in the trajectory is the last frame seen.
            for j, frame_idx in enumerate(range(idx_last_seen+1, i)):
                new_frame_id = initial_frame_id + frame_idx
                old_frame = frames[frame_idx]
                old_frame_agents = old_frame.all_agents

                new_frame = ip.data.episode.Frame(old_frame.time, old_frame.dead_ids)

                for agent_id, agent in old_frame_agents.items():
                    if agent_id == aid:
                        agent = AgentState(new_frame_id, new_path[j], new_vel[j], new_acc[j], new_heading[j], metadata)
                    new_frame.add_agent_state(agent_id, agent)

                occlusion_frames.append(new_frame)

        idx_last_seen = i

        # Add the frame in which the target is visible again.
        occlusion_frames.append(frame)
    return occlusion_frames

def dump_results(objects, name: str):
    """Saves results binary"""
    filename = name + '.pkl'
    filename = get_igp2_results_dir() + f"/{filename}"

    with open(filename, 'wb') as f:
        dill.dump(objects, f)


# Replace ProcessPoolExecutor with this for debugging without parallel execution
class MockProcessPoolExecutor():
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def submit(self, fn, *args, **kwargs):
        # execute functions in series without creating threads
        # for easier unit testing
        result = fn(*args, **kwargs)
        return result

    def shutdown(self, wait=True):
        pass


if __name__ == '__main__':

    set_working_dir()
    config = create_args()
    scenario_name = config['scenario']
    experiment_name = scenario_name + config['output']
    logger = setup_logging(level=logging.INFO, log_dir="baselines/igp2/logs", log_name=experiment_name)

    SCENARIOS = [scenario_name]

    DATASET = config["dataset"]
    if DATASET not in ('test', 'valid'):
        logger.error("Invalid dataset specified")
        sys.exit(1)

    TUNING = config["tuning"]
    if TUNING not in (0, 1):
        logger.error("Invalid tuning argument specified, use 0 or 1.")
        sys.exit(1)
    TUNING = bool(TUNING)

    REWARD_AS_DIFFERENCE = config["reward_scheme"]
    if REWARD_AS_DIFFERENCE not in (0, 1):
        logger.error("Invalid reward_scheme argument specified, use 0 or 1.")
        sys.exit(1)
    REWARD_AS_DIFFERENCE = bool(REWARD_AS_DIFFERENCE)

    max_workers = None if config['num_workers'] == 0 else config['num_workers']
    if max_workers is not None and max_workers <= 0:
        logger.error("Specify a valid number of workers or leave to default")
        sys.exit(1)

    results = []

    if TUNING:
        cost_factors_arr = []
        cost_factors_arr.append(
            {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 1000, "angular_velocity": 0.0,
             "angular_acceleration": 0., "curvature": 0.0, "safety": 0.})
        cost_factors_arr.append(
            {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 0.1, "angular_velocity": 0.0,
             "angular_acceleration": 0., "curvature": 0.0, "safety": 0.})
        cost_factors_arr.append(
            {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 1, "angular_velocity": 0.0,
             "angular_acceleration": 0., "curvature": 0.0, "safety": 0.})
        cost_factors_arr.append(
            {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 10, "angular_velocity": 0.0,
             "angular_acceleration": 0., "curvature": 0.0, "safety": 0.})
        cost_factors_arr.append(
            {"time": 0.001, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 100, "angular_velocity": 0.0,
             "angular_acceleration": 0., "curvature": 0.0, "safety": 0.})
        for idx, cost_factors in enumerate(cost_factors_arr):
            logger.info(f"Starting experiment {idx} with cost factors {cost_factors}.")
            t_start = time.perf_counter()
            result_experiment = run_experiment(cost_factors, max_workers=max_workers)
            results.append(copy.deepcopy(result_experiment))
            t_end = time.perf_counter()
            logger.info(f"Experiment {idx} completed in {t_end - t_start} seconds.")
    else:
        logger.info(f"Starting experiment")
        t_start = time.perf_counter()
        result_experiment = run_experiment(cost_factors=None, max_workers=max_workers)
        results.append(copy.deepcopy(result_experiment))
        t_end = time.perf_counter()
        logger.info(f"Experiment completed in {t_end - t_start} seconds.")

    dump_results(results, experiment_name)
    logger.info("All experiments complete.")