import json
from typing import Dict, List

import pandas as pd
import numpy as np
import math
from igp2 import AgentState, Box
from igp2.data import Episode
from igp2.data.scenario import InDScenario, ScenarioConfig
from igp2.opendrive.map import Map

from ogrit.core.feature_extraction import FeatureExtractor, GoalDetector
from shapely.geometry import LineString
from shapely.errors import TopologicalError

from ogrit.core.base import get_data_dir, get_base_dir, get_scenarios_dir

FRAME_STEP_SIZE = 25  # take a frame every 25 in the original episode frames (i.e., one per second)


def load_dataset_splits():
    with open(get_base_dir() + '/ogrit/core/dataset_split.json', 'r') as f:
        return json.load(f)


def get_dataset(scenario_name, subset='train', features=True):

    data_set_splits = load_dataset_splits()
    episode_idxes = data_set_splits[scenario_name][subset]
    episode_training_sets = []

    for episode_idx in episode_idxes:
        episode_training_set = pd.read_csv(
            get_data_dir() + '{}_e{}.csv'.format(scenario_name, episode_idx, subset))
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


def get_multi_scenario_dataset(scenario_names: List[str], subset='train') -> pd.DataFrame:
    scenario_datasets = []
    for scenario_idx, scenario_name in enumerate(scenario_names):
        scenario_dataset = get_dataset(scenario_name, subset)
        scenario_dataset['scenario'] = scenario_idx
        scenario_datasets.append(scenario_dataset)
    dataset = pd.concat(scenario_datasets)
    return dataset


def get_goal_priors(training_set, goal_types, alpha=0):
    agent_goals = training_set[['episode', 'agent_id', 'true_goal', 'true_goal_type']].drop_duplicates()
    print('training_vehicles: {}'.format(agent_goals.shape[0]))
    goal_counts = pd.DataFrame(data=[(x, t, 0) for x in range(len(goal_types)) for t in goal_types[x]],
                               columns=['true_goal', 'true_goal_type', 'goal_count'])

    goal_counts = goal_counts.set_index(['true_goal', 'true_goal_type'])
    goal_counts['goal_count'] += agent_goals.groupby(['true_goal', 'true_goal_type']).size()
    goal_counts = goal_counts.fillna(0)

    goal_priors = ((goal_counts.goal_count + alpha) / (agent_goals.shape[0] + alpha * goal_counts.shape[0])).rename('prior')
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


def get_trajectories(scenario, episode, trimmed=False):
    goals = {}  # key: agent id, value: goal idx
    trimmed_trajectories = {}

    # detect goal, and trim trajectory past the goal
    goal_detector = GoalDetector(scenario.config.goals)
    for agent_id, agent in episode.agents.items():
        if agent.metadata.agent_type in ['car', 'truck_bus']:
            agent_goals, goal_frame_idxes = goal_detector.detect_goals(agent.trajectory)
            if len(agent_goals) > 0:

                if trimmed:
                    end_idx = min(goal_frame_idxes)
                    trimmed_trajectory = agent.trajectory.slice(0, end_idx)
                else:
                    trimmed_trajectory = agent.trajectory
                goals[agent_id] = agent_goals[-1]
                trimmed_trajectories[agent_id] = trimmed_trajectory
    return trimmed_trajectories, goals


def get_trajectory_reachable_goals(trajectory, feature_extractor, scenario):
    # iterate through each sampled point in time for trajectory
    reachable_goals_list = []
    # get reachable goals at each timestep until there is only 1 possible goal.
    for idx in range(0, len(trajectory.path)):
        typed_goals = feature_extractor.get_typed_goals(trajectory.slice(0, idx + 1), scenario.config.goals)
        if len([r for r in typed_goals if r is not None]) > 1:
            reachable_goals_list.append(typed_goals)
        else:
            break
    return reachable_goals_list


def get_first_last_frame_ids(episode, vehicle_id):
    initial_frame_id = episode.agents[vehicle_id].metadata.initial_time
    final_frame_id = episode.agents[vehicle_id].metadata.final_time
    return initial_frame_id, final_frame_id


def _get_frame_ids(episode, target_agent_id, ego_agent_id=None):
    """
    If the ego agent id is given, return the ids of the frames in which both the ego and target are alive.
    Otherwise, return the ids of the frames in which the target is alive.
    """
    if ego_agent_id is not None:

        # Get the frames in which both the ego and the target vehicles are alive.
        initial_frame_id_target, last_frame_id_target = get_first_last_frame_ids(episode, target_agent_id)
        initial_frame_id_ego, last_frame_id_ego = get_first_last_frame_ids(episode, ego_agent_id)
        initial_frame_id = max(initial_frame_id_target, initial_frame_id_ego)
        last_frame_id = min(last_frame_id_target, last_frame_id_ego)

        # Only take samples in which the two vehicles are alive for at least FRAME_STEP_SIZE number of frames.
        if last_frame_id_ego - initial_frame_id_ego < FRAME_STEP_SIZE or initial_frame_id > last_frame_id:
            return None

        # Return the time steps in the target's trajectory in which both the target and ego are alive.
        start_trajectory_idx = initial_frame_id - initial_frame_id_target
        end_trajectory_idx = start_trajectory_idx + min(last_frame_id, last_frame_id_target) - initial_frame_id

    else:
        initial_frame_id, last_frame_id = get_first_last_frame_ids(episode, target_agent_id)
        initial_frame_id_target = initial_frame_id
        start_trajectory_idx = 0
        end_trajectory_idx = math.inf

    return initial_frame_id_target, initial_frame_id, last_frame_id, start_trajectory_idx, end_trajectory_idx


def is_target_vehicle_occluded(current_frame_id, feature_extractor, target_agent_id, ego_agent_id, episode_frames):
    occlusion_frame_id = math.ceil(current_frame_id / FRAME_STEP_SIZE)
    frame_occlusions = feature_extractor.occlusions[occlusion_frame_id]

    occlusions = frame_occlusions[ego_agent_id]

    target_agent = episode_frames[current_frame_id][target_agent_id]

    # Get the current lane on which the target vehicle is.
    try:
        lane_on = feature_extractor.scenario_map.lanes_at(target_agent.position)[0]
    except IndexError:
        # Treat the vehicle as occluded since it is outside any lane.
        return True

    # Get the occlusions on the lane the target is on.
    lane_occlusion = occlusions[lane_on.parent_road.id][lane_on.id]

    if lane_occlusion is None:
        # There are no occlusions on that lane.
        return False

    vehicle_boundary = LineString(get_vehicle_boundary(target_agent)).buffer(0.001)

    # If the vehicle is in the occluded area, it is missing.
    return lane_occlusion.contains(vehicle_boundary)


def extract_samples(feature_extractor, scenario, episode, extract_missing_features=False):

    episode_frames = get_episode_frames(episode)
    trajectories, goals = get_trajectories(scenario, episode, trimmed=not extract_missing_features)

    samples_list = []

    for target_agent_idx, (target_agent_id, trajectory) in enumerate(trajectories.items()):
        print('target agent {}/{}'.format(target_agent_idx, len(trajectories) - 1))

        # Get all the reachable goals at every time step of the trajectory, until there is only 1 goal left.
        full_reachable_goals_list = get_trajectory_reachable_goals(trajectory, feature_extractor, scenario)

        # For how many time steps is the target vehicle alive.
        target_lifespan = len(full_reachable_goals_list)

        for ego_agent_idx, (ego_agent_id, _) in enumerate(trajectories.items()):
            if ego_agent_id == target_agent_id or episode.agents[ego_agent_id].parked():
                continue

            # If we don't consider occlusions, we don't need the ego vehicle. We thus run the rest of the code once.
            if not extract_missing_features and ego_agent_idx != 0:
                break

            ids_goals = _get_frame_ids(episode, target_agent_id,
                                       ego_agent_id if extract_missing_features else None)

            if ids_goals is None:
                # We have no frames in which both vehicles are alive at the same time.
                continue

            target_initial, initial_frame_id, final_frame_id, start_trajectory_idx, end_trajectory_idx = ids_goals

            if extract_missing_features:
                # Get the target vehicle's possible goals in the time steps in which both the ego and the target
                # are alive.
                max_timestep = min(end_trajectory_idx+1, target_lifespan)
                reachable_goals_list = full_reachable_goals_list[start_trajectory_idx:max_timestep]
            else:
                reachable_goals_list = full_reachable_goals_list

            true_goal_idx = goals[target_agent_id]

            if reachable_goals_list and reachable_goals_list[0][true_goal_idx] is not None:

                # get true goal
                true_goal_route = reachable_goals_list[0][true_goal_idx].lane_path
                true_goal_type = feature_extractor.goal_type(true_goal_route)

                # Align the frames with those for which we have occlusions (one every second).
                initial_frame_offset = FRAME_STEP_SIZE * math.ceil(initial_frame_id/FRAME_STEP_SIZE) - initial_frame_id
                # Save the first frame in which the target vehicle wasn't occluded w.r.t the ego.
                first_frame_target_not_occluded = None

                for idx in range(initial_frame_offset, len(reachable_goals_list)+1, FRAME_STEP_SIZE):

                    try:
                        reachable_goals = reachable_goals_list[idx]
                    except IndexError:
                        # There is no goal recognition to perform at this time step.
                        continue

                    current_frame_id = initial_frame_id + idx

                    if current_frame_id > final_frame_id:
                        break

                    # Don't include the frames in which the target vehicle is occluded w.r.t the ego.
                    if extract_missing_features:

                        if is_target_vehicle_occluded(current_frame_id, feature_extractor, target_agent_id,
                                                      ego_agent_id, episode_frames):
                            continue

                        if first_frame_target_not_occluded is None:
                            first_frame_target_not_occluded = episode_frames[current_frame_id]

                    # Take the frames of what the ego has seen from the moment both the ego and target became alive.
                    frames = episode_frames[initial_frame_id:current_frame_id + 1]

                    # iterate through each goal for that point in time.
                    for goal_idx, typed_goal in enumerate(reachable_goals):
                        if typed_goal is not None:

                            if extract_missing_features:

                                try:
                                    features = feature_extractor.extract(target_agent_id, frames, typed_goal,
                                                                         ego_agent_id=ego_agent_id,
                                                                         initial_frame=first_frame_target_not_occluded)
                                except TopologicalError:
                                    continue

                            else:
                                features = feature_extractor.extract(target_agent_id, frames, typed_goal)

                            sample = features.copy()
                            sample['agent_id'] = target_agent_id
                            sample['ego_agent_id'] = ego_agent_id
                            sample['possible_goal'] = goal_idx
                            sample['true_goal'] = true_goal_idx
                            sample['true_goal_type'] = true_goal_type
                            sample['frame_id'] = current_frame_id
                            sample['initial_frame_id'] = target_initial
                            sample['fraction_observed'] = (current_frame_id - target_initial) / target_lifespan

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
    scenario_name, episode_idx, extract_indicator_features = params

    print('scenario {} episode {}'.format(scenario_name, episode_idx))

    scenario_map = Map.parse_from_opendrive(get_scenarios_dir() + f"maps/{scenario_name}.xodr")
    scenario_config = ScenarioConfig.load(get_scenarios_dir() + f"configs/{scenario_name}.json")
    scenario = InDScenario(scenario_config)

    if extract_indicator_features:
        feature_extractor = FeatureExtractor(scenario_map, scenario_name, episode_idx)
    else:
        feature_extractor = FeatureExtractor(scenario_map)

    episode = scenario.load_episode(episode_idx)

    samples = extract_samples(feature_extractor, scenario, episode, extract_indicator_features)
    samples.to_csv(get_data_dir() + '{}_e{}.csv'.format(scenario_name, episode_idx), index=False)
    print('finished scenario {} episode {}'.format(scenario_name, episode_idx))
