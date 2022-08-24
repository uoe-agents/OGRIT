import copy
import pandas as pd
import pickle
import dill
import concurrent.futures
import argparse
import sys
import time

import igp2 as ip
from igp2 import setup_logging, AgentState
from igp2.data.data_loaders import InDDataLoader
from igp2.goal import PointGoal
from scipy.interpolate import CubicSpline
from shapely.geometry import Point, LineString
from igp2.recognition.goalrecognition import *
from igp2.recognition.astar import AStar
from igp2.cost import Cost
from igp2.results import *
from igp2.planlibrary.maneuver import Maneuver, SwitchLane
from igp2.planlibrary.macro_action import ChangeLane

from ogrit.core.base import set_working_dir, get_occlusions_dir

# todo: code from https://github.com/uoe-agents/IGP2/blob/main/scripts/experiments/experiment_multi_process.py
def create_args():
    # TODO: modify folders name in description
    config_specification = argparse.ArgumentParser(description="""
This experiment will perform goal prediction for all agents and frames specified 
by a .csv file located in scripts/experiments/data/evaluation_set, following the
naming convention "scenario"_e"episode_no".csv, where episode_no is defined as the 
index of the recording ids defined in the json files. \n
A result binary will be generated and stored in scripts/experiments/data/results \n
Logs can be accessed in scripts/experiments/data/logs \n
Make sure to create these folders ahead of running the script.
     """, formatter_class=argparse.RawTextHelpFormatter)

    config_specification.add_argument('--num_workers', default="0",
                                      help="Number of parralel processes. Set 0 for auto", type=int)
    config_specification.add_argument('--output', default="experiment",
                                      help="Output .pkl filename", type=str)
    config_specification.add_argument('--tuning', default="0",
                                      help="0: runs default tuning parameters defined in the scenario JSON. \
                                      1: runs all cost parameters defined in the script.", type=int)
    config_specification.add_argument('--reward_scheme', default="1",
                                      help="0: runs the default reward scheme defined in the IGP2 paper. \
                                      1: runs alternate reward scheme defined in Cost.cost_difference_resampled().",
                                      type=int)
    config_specification.add_argument('--dataset', default="valid",
                                      help="valid: runs on the validation dataset, test: runs on the test dataset",
                                      type=str)

    parsed_config_specification = vars(config_specification.parse_args())
    return parsed_config_specification


def extract_goal_data(goals_data):
    """Creates a list of Goal objects from the .json scenario file data"""
    goals = []
    for goal_data in goals_data:
        point = Point(np.array(goal_data))
        goals.append(PointGoal(point, 2.))

    return goals


def read_and_process_data(scenario, episode_id): # todo: use this instead
    """Identifies which frames and agents to perform goal recognition for in episode,
    from a provided .csv file."""

    file_path = f'data/{scenario}_e{episode_id}.csv'
    data = pd.read_csv(file_path).drop_duplicates(['frame_id', 'agent_id', 'ego_agent_id'])
    # todo: remove duplicates
    return data


def goal_recognition_agent(frames, recordingID, framerate, aid, ego_id, data, goal_recognition: GoalRecognition,
                           goal_probabilities: GoalsProbabilities):
    """Computes the goal recognition results for specified agent at specified frames."""
    goal_probabilities_c = copy.deepcopy(goal_probabilities)
    result_agent = None
    for frame in frames:
        if aid in frame.dead_ids:
            frame.dead_ids.remove(aid)

    frame_ini = min(data['frame_id'])

    for _, row in data.iterrows():
        try:
            if result_agent is None:
                result_agent = AgentResult(row['true_goal'])
            frame_id = row['frame_id']
            agent_states = [frame.agents[aid] for frame in frames[0:frame_id - frame_ini + 1]] # todo: replace where frames come from
            trajectory = ip.StateTrajectory(framerate, frames=agent_states)
            t_start = time.perf_counter()
            goal_recognition.update_goals_probabilities(goal_probabilities_c, trajectory, aid,
                                                        frame_ini=frames[0].agents,
                                                        frame=frames[frame_id - frame_ini].agents, maneuver=None)
            t_end = time.perf_counter()
            result_agent.add_data((frame_id, copy.deepcopy(goal_probabilities_c), t_end - t_start, trajectory.path[-1]))
        except Exception as e:
            logger.error(f"Fatal in recording_id: {recordingID} for aid: {aid} at frame {frame_id}.")
            logger.error(f"Error message: {str(e)}")

    return aid, result_agent


def multi_proc_helper(arg_list):
    """Allows to pass multiple arguments as multiprocessing routine."""
    return goal_recognition_agent(arg_list[0], arg_list[1], arg_list[2], arg_list[3], arg_list[4], arg_list[5],
                                  arg_list[6], arg_list[7])


def run_experiment(cost_factors: Dict[str, float] = None, use_priors: bool = True, max_workers: int = None):
    """Run goal prediction in parralel for each agent, across all specified scenarios."""
    result_experiment = ExperimentResult()

    for scenario_name in SCENARIOS:
        scenario_map = ip.Map.parse_from_opendrive(f"scenarios/maps/{scenario_name}.xodr")
        data_loader = InDDataLoader(f"scenarios/configs/{scenario_name}.json", [DATASET])
        data_loader.load()

        # Scenario specific parameters
        SwitchLane.TARGET_SWITCH_LENGTH = data_loader.scenario.config.target_switch_length
        ChangeLane.CHECK_ONCOMING = data_loader.scenario.config.check_oncoming
        ip.Trajectory.VELOCITY_STOP = 1.  # TODO make .json parameter
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

        # todo convert the goals from list to points
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
                f"Starting experiment in scenario: {scenario_name}, episode_id: {episode_ids[ind_episode]}, recording_id: {recordingID}")
            smoother = ip.VelocitySmoother(vmin_m_s=ip.Trajectory.VELOCITY_STOP, vmax_m_s=episode.metadata.max_speed, n=10,
                                        amax_m_s2=5, lambda_acc=10)
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
                frames = _get_occlusion_frames(frames, occlusions, aid, ego_id, goal_recognition)
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


def _get_occlusion_frames(frames, occlusions, aid, ego_id, goal_recognition):
    # todo: description. Given

    from ogrit.occlusion_detection.visualisation_tools import get_box  # todo

    idx_last_seen = 0
    occlusion_frames = []
    initial_frame_id = frames[0].time
    metadata = frames[0].agents[aid].metadata

    # todo: frames that take into account occlusions
    for i, frame in enumerate(frames):
        frame_id = frame.time
        ego_occlusions = occlusions[frame_id][ego_id]["occlusions"]
        target_occluded = LineString(get_box(frame.agents[aid]).boundary).buffer(0.001).within(ego_occlusions) # todo: could make it function

        nr_occluded_frames = i-idx_last_seen

        # If the target was visible in the previous frame, do nothing. Else, use a* to fill in the occluded part.
        if target_occluded:
            continue

        if idx_last_seen == i-1 or idx_last_seen == 0:
            idx_last_seen = i
        else:

            state_trajectory = ip.StateTrajectory(25, [frames[idx_last_seen].agents[aid]]) # todo: make 25 a variable `framerate`
            # Generate trajectory for those frames in which the target was occluded.
            trajectory, _ = goal_recognition.generate_trajectory(n_trajectories=1,
                                                                 agent_id=aid,
                                                                 frame=frames[idx_last_seen].agents,
                                                                 goal=PointGoal(frame.agents[aid].position, 2),
                                                                 state_trajectory=state_trajectory,
                                                                 n_resample=nr_occluded_frames)

            """
            smap = ip.Map.parse_from_opendrive(f"scenarios/maps/heckstrasse.xodr")
            ip.plot_map(smap)
            plt.plot(*frames[idx_last_seen].agents[aid].position, marker=".")
            plt.plot(*frame.agents[aid].position, marker="x")
            

            for traj in trajectory:
                plt.plot(*list(zip(*traj.path)), color="r")
            plt.show()
            """

            trajectory = trajectory[0]

            initial_heading = trajectory.heading[0]
            final_heading = trajectory.heading[-1]
            initial_direction = np.array([np.cos(initial_heading), np.sin(initial_heading)])
            final_direction = np.array([np.cos(final_heading), np.sin(final_heading)])

            cs_path = CubicSpline(trajectory.times, trajectory.path, bc_type=((1, initial_direction),
                                                                              (1, final_direction)))

            cs_velocity = CubicSpline(trajectory.times, trajectory.velocity)

            ts = np.linspace(0, trajectory.times[-1], nr_occluded_frames)

            new_path = cs_path(ts)
            new_vel = cs_velocity(ts)

            for j, frame_idx in enumerate(range(idx_last_seen+1, i), 1):
                new_frame_id = initial_frame_id + frame_idx
                old_frame_agents = frames[frame_idx].agents
                old_frame_agents[aid] = AgentState(new_frame_id,
                                                   new_path[j],
                                                   new_vel[j],
                                                   None,
                                                   None,
                                                   metadata)
                occlusion_frames.append(old_frame_agents)

        occlusion_frames.append(frame)
    return frames  # todo: return the updated frames

def dump_results(objects, name: str):
    """Saves results binary"""
    filename = name + '.pkl'
    foldername = 'baselines/igp2/results/'
    filename = foldername + filename

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
    experiment_name = config['output']
    logger = setup_logging(level=logging.INFO, log_dir="baselines/igp2/logs", log_name=experiment_name)

    SCENARIOS = ["heckstrasse"] # todo: add all scenarios

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
            result_experiment = run_experiment(cost_factors, use_priors=True, max_workers=max_workers)
            results.append(copy.deepcopy(result_experiment))
            t_end = time.perf_counter()
            logger.info(f"Experiment {idx} completed in {t_end - t_start} seconds.")
    else:
        logger.info(f"Starting experiment")
        t_start = time.perf_counter()
        result_experiment = run_experiment(cost_factors=None, use_priors=True, max_workers=max_workers)
        results.append(copy.deepcopy(result_experiment))
        t_end = time.perf_counter()
        logger.info(f"Experiment completed in {t_end - t_start} seconds.")

    dump_results(results, experiment_name)
    logger.info("All experiments complete.")