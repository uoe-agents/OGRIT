import json
import math
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd
from igp2 import AgentState, Box
from igp2.data import Episode
from igp2.data.scenario import InDScenario, ScenarioConfig
from igp2.opendrive.map import Map
from shapely.geometry import LineString
from tqdm import tqdm

from ogrit.core.base import get_base_dir, set_working_dir, get_result_file_path, \
    get_map_path, get_map_configs_path
from ogrit.core.feature_extraction import FeatureExtractor, GoalDetector
from ogrit.core.logger import logger


def load_dataset_splits():
    with open(get_base_dir() + '/ogrit/core/dataset_split.json', 'r') as f:
        return json.load(f)


def get_dataset(scenario_name, subset='train', features=True, update_hz=25):
    data_set_splits = load_dataset_splits()
    episode_idxes = data_set_splits[scenario_name][subset]
    episode_training_sets = []

    for episode_idx in episode_idxes:
        episode_training_set = pd.read_csv(get_result_file_path(scenario_name, update_hz, episode_idx))
        episode_training_set['episode'] = episode_idx
        episode_training_sets.append(episode_training_set)
    training_set = pd.concat(episode_training_sets)

    if features:
        return training_set
    else:
        unique_training_samples = training_set[['episode', 'agent_id', 'initial_frame_id', 'frame_id',
                                                'true_goal', 'true_goal_type', 'fraction_observed']
        ].drop_duplicates().reset_index()
        return unique_training_samples


def get_multi_scenario_dataset(scenario_names: List[str], subset='train', update_hz=25) -> pd.DataFrame:
    scenario_datasets = []
    for scenario_idx, scenario_name in enumerate(scenario_names):
        scenario_dataset = get_dataset(scenario_name, subset=subset, update_hz=update_hz)
        scenario_dataset['scenario'] = scenario_idx
        scenario_datasets.append(scenario_dataset)
    dataset = pd.concat(scenario_datasets)
    return dataset


def get_goal_priors(training_set, goal_types, alpha=0):
    agent_goals = training_set[['episode', 'agent_id', 'true_goal', 'true_goal_type']].drop_duplicates()
    logger.info('training_vehicles: {}'.format(agent_goals.shape[0]))
    goal_counts = pd.DataFrame(data=[(x, t, 0) for x in range(len(goal_types)) for t in goal_types[x]],
                               columns=['true_goal', 'true_goal_type', 'goal_count'])

    goal_counts = goal_counts.set_index(['true_goal', 'true_goal_type'])
    goal_counts['goal_count'] += agent_goals.groupby(['true_goal', 'true_goal_type']).size()
    goal_counts = goal_counts.fillna(0)

    goal_priors = ((goal_counts.goal_count + alpha) / (agent_goals.shape[0] + alpha * goal_counts.shape[0])).rename(
        'prior')
    goal_priors = goal_priors.reset_index()
    return goal_priors


def get_episode_frames(episode: Episode, exclude_parked_cars=True, exclude_bicycles=False, step=1) \
        -> List[Dict[int, AgentState]]:
    """
    Get all the frames in an episode, while removing pedestrians and possibly bicycles and parked cars

    Args:
        episode: Episode for which we want the frames.
        exclude_parked_cars: True if we don't want to get the parked vehicles in the frames.
        exclude_bicycles: True if we don't want the bicycles in the frames.
        step: Only return a frame every `step` frames in the episode.
    """
    episode_frames = []

    for i, frame in enumerate(episode.frames):
        if i % step != 0:
            continue

        new_frame = {}
        for agent_id, state in frame.agents.items():
            agent = episode.agents[agent_id]

            if not ((agent.parked() and exclude_parked_cars)
                    or agent.metadata.agent_type == 'pedestrian'
                    or (agent.metadata.agent_type == 'bicycle'
                        and exclude_bicycles)):
                new_frame[agent_id] = state
        episode_frames.append(new_frame)

    return episode_frames


def get_trimmed_trajectories(scenario, episode):
    goals = {}  # key: agent id, value: goal idx
    trimmed_trajectories = {}

    # detect goal, and trim trajectory past the goal
    goal_detector = GoalDetector(scenario.config.goals, scenario.config.goal_threshold)
    for agent_id, agent in episode.agents.items():
        if agent.metadata.agent_type in ['car', 'truck_bus']:
            agent_goals, goal_frame_idxes = goal_detector.detect_goals(agent.trajectory)
            if len(agent_goals) > 0:
                trimmed_trajectory = agent.trajectory.slice(0,
                                                            goal_frame_idxes[-1] + 1)  # add the +1 since we're slicing

                goals[agent_id] = agent_goals[-1]
                trimmed_trajectories[agent_id] = trimmed_trajectory
    return trimmed_trajectories, goals


@lru_cache(maxsize=128)
def get_trajectory_reachable_goals(trajectory, feature_extractor, scenario):
    # iterate through each sampled point in time for trajectory
    reachable_goals_list = []
    # get reachable goals at each timestep until there is only 1 possible goal.
    for idx in range(0, len(trajectory.path)):
        typed_goals = feature_extractor.get_typed_goals(trajectory.slice(0, idx + 1), scenario.config.goals,
                                                        scenario.config.goal_threshold)
        if len([r for r in typed_goals if r is not None]) > 0:
            reachable_goals_list.append(typed_goals)
        else:
            break
    return reachable_goals_list


def get_first_last_frame_ids(episode, vehicle_id):
    initial_frame_id = int(episode.agents[vehicle_id].metadata.initial_time)
    final_frame_id = int(episode.agents[vehicle_id].metadata.final_time)
    return initial_frame_id, final_frame_id


def _get_frame_ids(episode, target_agent_id, update_hz, ego_agent_id=None):
    f"""
    If the ego agent id is given, return the ids of the frames in which both the ego and target are alive.
    Otherwise, return the ids of the frames in which the target is alive.

    Return:
        initial_frame_id_target: id of the first frame in which the target is alive. None if the ego id is given and 
                                 there are no frames in which both the target and ego are alive.
        initial_frame_id:        if of the first frame in which both the target and the ego are alive. If the ego is 
                                 not given, this is the same as initial_frame_id_target
        last_frame_id:           if of the last frame in which both the target and the ego are alive. If the ego is not 
                                 given, this is the same as the last frame in which the target is alive.
    """
    if ego_agent_id is not None:

        # Get the frames in which both the ego and the target vehicles are alive.
        initial_frame_id_target, last_frame_id_target = get_first_last_frame_ids(episode, target_agent_id)
        initial_frame_id_ego, last_frame_id_ego = get_first_last_frame_ids(episode, ego_agent_id)

        initial_frame_id = max(initial_frame_id_target, initial_frame_id_ego)
        last_frame_id = min(last_frame_id_target, last_frame_id_ego)

        # Only take samples in which the two vehicles are alive for at least update_hz number of frames.
        if last_frame_id_ego - initial_frame_id_ego < update_hz or initial_frame_id > last_frame_id:
            return None, None, None

    else:
        initial_frame_id, last_frame_id = get_first_last_frame_ids(episode, target_agent_id)
        initial_frame_id_target = initial_frame_id

    return initial_frame_id_target, initial_frame_id, last_frame_id


def is_target_vehicle_occluded(current_frame_id, occlusions, target_agent_id, ego_agent_id, episode_frames):
    occlusions = occlusions[current_frame_id][ego_agent_id]["occlusions"]

    target_agent = episode_frames[current_frame_id][target_agent_id]
    vehicle_boundary = LineString(get_vehicle_boundary(target_agent)).buffer(0.001)

    # If the vehicle is in the occluded area, it is missing.
    return occlusions.contains(vehicle_boundary)


def extract_samples(feature_extractor, scenario, episode, update_hz, extract_missing_features=False):
    episode_frames = get_episode_frames(episode)
    trajectories, goals = get_trimmed_trajectories(scenario, episode)

    samples_list = []

    for target_agent_idx, (target_agent_id, trajectory) in enumerate(tqdm(trajectories.items())):

        # Get all the reachable goals at every time step of the trajectory.
        full_reachable_goals_list = get_trajectory_reachable_goals(trajectory, feature_extractor, scenario)

        target_lifespan = len(trajectory.timesteps)

        for ego_agent_idx, (ego_agent_id, _) in enumerate(trajectories.items()):
            if ego_agent_id == target_agent_id or episode.agents[ego_agent_id].parked():
                continue

            # If we don't consider occlusions, we don't need the ego vehicle. We thus run the rest of the code once.
            if not extract_missing_features and ego_agent_idx != 0:
                break

            target_initial_frame, initial_frame_id, final_frame_id = _get_frame_ids(episode, target_agent_id, update_hz,
                                                                                    ego_agent_id if
                                                                                    extract_missing_features else None)
            if target_initial_frame is None:
                # We have no frames in which both vehicles are alive at the same time.
                continue

            if extract_missing_features:
                # Get the target vehicle's possible goals in the time steps in which both the ego and the target
                # are alive.
                start_trajectory_idx = initial_frame_id - target_initial_frame
                end_trajectory_idx = final_frame_id - target_initial_frame
                reachable_goals_list = full_reachable_goals_list[start_trajectory_idx:end_trajectory_idx + 1]
            else:
                reachable_goals_list = full_reachable_goals_list

            true_goal_idx = goals[target_agent_id]

            # Check the agent can reach some goal and, in particular, the true goal.
            if reachable_goals_list and reachable_goals_list[0][true_goal_idx] is not None:

                # get true goal
                true_goal_route = reachable_goals_list[0][true_goal_idx].lane_path
                true_goal_type = feature_extractor.goal_type(true_goal_route)

                # Align the frames so that they are multiples of update_hz.
                initial_frame_offset = update_hz * math.ceil(
                    initial_frame_id / update_hz) - initial_frame_id
                # Save the first frame in which the target vehicle wasn't occluded w.r.t the ego.
                first_frame_target_not_occluded = None

                target_occlusion_history = []  # booleans indicating whether the target was occluded in each frame

                # Get a sample every update_hz time steps from when the target first becomes visible to the ego
                # until it is last visible.
                for step_idx in range(initial_frame_offset, len(reachable_goals_list), update_hz):

                    reachable_goals = reachable_goals_list[step_idx]
                    current_frame_id = initial_frame_id + step_idx

                    # Don't include the frames in which the target vehicle is occluded w.r.t the ego.
                    if extract_missing_features:

                        target_occluded = is_target_vehicle_occluded(current_frame_id, feature_extractor.occlusions,
                                                                     target_agent_id, ego_agent_id, episode_frames)
                        target_occlusion_history.append(target_occluded)
                        if target_occluded:
                            continue

                        if first_frame_target_not_occluded is None:
                            first_frame_target_not_occluded = episode_frames[current_frame_id]

                    # Take the frames of what the ego has seen from the moment both the ego and target became alive.
                    frames = episode_frames[target_initial_frame:current_frame_id + 1]

                    # iterate through each goal for that point in time.
                    for goal_idx, typed_goal in enumerate(reachable_goals):
                        if typed_goal is not None:

                            if extract_missing_features:

                                features = feature_extractor.extract(target_agent_id, frames, typed_goal,
                                                                     ego_agent_id=ego_agent_id,
                                                                     initial_frame=first_frame_target_not_occluded,
                                                                     target_occlusion_history=target_occlusion_history,
                                                                     fps=update_hz)
                            else:
                                features = feature_extractor.extract(target_agent_id, frames, typed_goal, fps=update_hz)

                            sample = features.copy()
                            sample['agent_id'] = target_agent_id

                            if extract_missing_features:
                                sample['ego_agent_id'] = ego_agent_id
                            sample['possible_goal'] = goal_idx
                            sample['true_goal'] = true_goal_idx
                            sample['true_goal_type'] = true_goal_type
                            sample['delta_x_from_possible_goal'] = abs(typed_goal.goal.center.x - features['x'])
                            sample['delta_y_from_possible_goal'] = abs(typed_goal.goal.center.y - features['y'])
                            sample['frame_id'] = current_frame_id
                            sample['initial_frame_id'] = target_initial_frame
                            sample['fraction_observed'] = (current_frame_id - target_initial_frame) / target_lifespan

                            samples_list.append(sample)

    samples = pd.DataFrame(data=samples_list)
    return samples


def get_vehicle_boundary(vehicle):
    return Box(np.array([vehicle.position[0],
                         vehicle.position[1]]),
               vehicle.metadata.length,
               vehicle.metadata.width,
               vehicle.heading).boundary


def prepare_episode_dataset(params):
    scenario_name, episode_idx, extract_indicator_features, update_hz = params

    # update_hz =  take a sample every update_hz frames in the original episode frames (e.g., if 25,
    # then take one frame per second)

    logger.info('scenario {} episode {}'.format(scenario_name, episode_idx))

    scenario_map = Map.parse_from_opendrive(get_map_path(scenario_name))
    scenario_config = ScenarioConfig.load(get_map_configs_path(scenario_name))

    set_working_dir()
    scenario = InDScenario(scenario_config)

    if extract_indicator_features:
        feature_extractor = FeatureExtractor(scenario_map, scenario_name, episode_idx)
    else:
        feature_extractor = FeatureExtractor(scenario_map)

    episode = scenario.load_episode(episode_idx)

    samples = extract_samples(feature_extractor, scenario, episode, update_hz,
                              extract_missing_features=extract_indicator_features)
    samples.to_csv(get_result_file_path(scenario_name, update_hz, episode_idx), index=False)
    logger.info(f'finished scenario {scenario_name} episode {episode_idx}')
