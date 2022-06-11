import json
from typing import Dict, List

import pandas as pd
import numpy as np
import math
from igp2 import AgentState, Box
from igp2.data import Episode
from igp2.data.scenario import InDScenario, ScenarioConfig
from igp2.opendrive.map import Map

from grit.core.feature_extraction import FeatureExtractor, GoalDetector
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from grit.core.base import get_data_dir, get_base_dir

FRAME_STEP_SIZE = 25  # take a frame every 25 in the original episode frames (i.e., one per second)


def load_dataset_splits():
    with open(get_base_dir() + '/grit/core/dataset_split.json', 'r') as f:
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
    Get all of the frames in an episode, while removing pedestrians and possibly bicycles and parked cars

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
    # get reachable goals at each timestep
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


def get_frame_ids_and_goals(scenario, episode, trajectory, target_agent_id, feature_extractor, ego_agent_id=None):

    if ego_agent_id:

        # Get the frames in which both the ego and the target vehicles are alive.
        initial_frame_id_target, last_frame_id_target = get_first_last_frame_ids(episode, target_agent_id)
        initial_frame_id_ego, last_frame_id_ego = get_first_last_frame_ids(episode, ego_agent_id)
        initial_frame_id = max(initial_frame_id_target, initial_frame_id_ego)
        last_frame_id = min(last_frame_id_target, last_frame_id_ego)

        # Only take samples in which the two vehicles are alive for at least FRAME_STEP_SIZE number of frames.
        if last_frame_id_ego - initial_frame_id_ego < FRAME_STEP_SIZE or initial_frame_id > last_frame_id:
            return None

        # For each of the time steps, take the goals that are reachable for the target.
        start_trajectory_idx = initial_frame_id - initial_frame_id_target
        end_trajectory_idx = start_trajectory_idx + min(last_frame_id, last_frame_id_target) - initial_frame_id
        reachable_goals_list = get_trajectory_reachable_goals(trajectory.slice(start_trajectory_idx,
                                                                               end_trajectory_idx + 1),
                                                              feature_extractor,
                                                              scenario)

    else:
        initial_frame_id, last_frame_id = get_first_last_frame_ids(episode, target_agent_id)
        reachable_goals_list = get_trajectory_reachable_goals(trajectory, feature_extractor, scenario)

    return initial_frame_id, last_frame_id, reachable_goals_list


def is_target_vehicle_occluded(current_frame_id, feature_extractor, target_agent_id, ego_agent_id, episode_frames):
    occlusion_frame_id = math.ceil(current_frame_id / FRAME_STEP_SIZE)
    frame_occlusions = feature_extractor.occlusions[str(occlusion_frame_id)]

    occlusions = get_vehicle_occlusions(frame_occlusions, ego_agent_id)

    target_agent = episode_frames[current_frame_id][target_agent_id]

    try:
        l = feature_extractor.scenario_map.lanes_at(target_agent.position)[0]
    except IndexError:
        # Treat it as if occluded, as the vehicle is outside any lane
        return True

    try:
        # We have an exception if there are no occlusions on that lane.
        lane_occlusion = occlusions[str(l.parent_road.id)][str(l.id)]
        lane_occlusion = unary_union([Polygon(list(zip(*xy))) for xy in lane_occlusion])

        vehicle_boundary = MultiPoint(get_vehicle_boundary(target_agent))

        # If the vehicle is inside the occluded area,
        if lane_occlusion.contains(vehicle_boundary):
            return True

    finally:
        return False


def extract_samples(feature_extractor, scenario, episode, extract_missing_features=False):

    episode_frames = get_episode_frames(episode)
    trajectories, goals = get_trajectories(scenario, episode, trimmed=not extract_missing_features)

    samples_list = []

    for target_agent_idx, (target_agent_id, trajectory) in enumerate(trajectories.items()):
        print('target agent {}/{}'.format(target_agent_idx, len(trajectories) - 1))

        for ego_agent_idx, (ego_agent_id, _) in enumerate(trajectories.items()):
            if ego_agent_id == target_agent_id:
                continue

            print('ego agent {}/{}'.format(ego_agent_idx, len(trajectories) - 1))

            # If we don't consider occlusions, we don't need the ego vehicle. We thus run the rest of the code once.
            if not extract_missing_features and ego_agent_idx != 0:
                break

            ids_goals = get_frame_ids_and_goals(scenario, episode, trajectory, target_agent_id, feature_extractor,
                                                ego_agent_id if extract_missing_features else None)

            if ids_goals is None:
                # We have no frames in which both vehicles are alive at the same time.
                continue

            initial_frame_id, final_frame_id, reachable_goals_list = ids_goals

            true_goal_idx = goals[target_agent_id]

            if reachable_goals_list and reachable_goals_list[0][true_goal_idx] is not None:

                # get true goal
                true_goal_route = reachable_goals_list[0][true_goal_idx].lane_path
                true_goal_type = feature_extractor.goal_type(true_goal_route)

                # Align the frames with those for which we have occlusions (one every second).
                initial_frame_offset = FRAME_STEP_SIZE * math.ceil(initial_frame_id/FRAME_STEP_SIZE) - initial_frame_id
                # Save the first frame in which the target vehicle wasn't occluded w.r.t the ego.
                first_frame_target_not_occluded_id = None

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

                        if first_frame_target_not_occluded_id is None:
                            first_frame_target_not_occluded_id = episode_frames[current_frame_id]

                    # Take the frames of what the ego has seen from the moment both the ego and target became alive.
                    frames = episode_frames[initial_frame_id:current_frame_id + 1]

                    # iterate through each goal for that point in time.
                    for goal_idx, typed_goal in enumerate(reachable_goals):
                        if typed_goal is not None:

                            if extract_missing_features:
                                features = feature_extractor.extract(target_agent_id, frames, typed_goal,
                                                                     ego_agent_id=ego_agent_id,
                                                                     initial_frame=episode_frames[initial_frame_id])
                            else:
                                features = feature_extractor.extract(target_agent_id, frames, typed_goal)

                            sample = features.copy()
                            sample['agent_id'] = target_agent_id
                            sample['possible_goal'] = goal_idx
                            sample['true_goal'] = true_goal_idx
                            sample['true_goal_type'] = true_goal_type
                            sample['frame_id'] = current_frame_id
                            sample['initial_frame_id'] = initial_frame_id
                            sample['fraction_observed'] = idx / len(reachable_goals_list)

                            samples_list.append(sample)

    samples = pd.DataFrame(data=samples_list)
    return samples


def get_vehicle_occlusions(frame_occlusions, ego_agent_id):
    occlusions = []
    for vehicle_occlusions in frame_occlusions:
        if vehicle_occlusions["ego_agent_id"] == ego_agent_id:
            occlusions = vehicle_occlusions["occlusions"]
    return occlusions


def get_vehicle_boundary(vehicle):
    return Box(np.array([vehicle.position[0],
                         vehicle.position[1]]),
               vehicle.metadata.length,
               vehicle.metadata.width,
               vehicle.heading).boundary


def prepare_episode_dataset(params):
    scenario_name, episode_idx, extract_missing_features = params
    print('scenario {} episode {}'.format(scenario_name, episode_idx))

    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{scenario_name}.xodr")
    scenario_config = ScenarioConfig.load(f"scenarios/configs/{scenario_name}.json")
    scenario = InDScenario(scenario_config)

    if extract_missing_features:
        feature_extractor = FeatureExtractor(scenario_map, scenario_name, episode_idx)
    else:
        feature_extractor = FeatureExtractor(scenario_map)

    episode = scenario.load_episode(episode_idx)

    samples = extract_samples(feature_extractor, scenario, episode, extract_missing_features)
    samples.to_csv(get_data_dir() + '{}_e{}.csv'.format(scenario_name, episode_idx), index=False)
    print('finished scenario {} episode {}'.format(scenario_name, episode_idx))
