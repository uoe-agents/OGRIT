import json
from typing import Dict, List

import pandas as pd
import math
from igp2 import AgentState
from igp2.data import Episode
from igp2.data.scenario import InDScenario, ScenarioConfig
from igp2.opendrive.map import Map

from grit.core.feature_extraction import FeatureExtractor, GoalDetector
from grit.occlusion_detection.missing_feature_extraction import MissingFeatureExtractor
from shapely.geometry import Point, Polygon, MultiPolygon

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


def get_trimmed_trajectories(scenario, episode):
    goals = {}  # key: agent id, value: goal idx
    trimmed_trajectories = {}

    # detect goal, and trim trajectory past the goal
    goal_detector = GoalDetector(scenario.config.goals)
    for agent_id, agent in episode.agents.items():
        if agent.metadata.agent_type in ['car', 'truck_bus']:
            agent_goals, goal_frame_idxes = goal_detector.detect_goals(agent.trajectory)
            if len(agent_goals) > 0:
                end_idx = min(goal_frame_idxes)
                trimmed_trajectory = agent.trajectory.slice(0, end_idx)
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
    initial_frame_id = episode.agents[vehicle_id].metadata.initial_time / FRAME_STEP_SIZE
    final_frame_id = episode.agents[vehicle_id].metadata.final_time / FRAME_STEP_SIZE
    return math.ceil(initial_frame_id), math.floor(final_frame_id)


def extract_samples(feature_extractor, scenario, episode, missing_feature_extractor=None):

    scenario_map = missing_feature_extractor.scenario_map
    episode_frames = get_episode_frames(episode, exclude_parked_cars=False, exclude_bicycles=True, step=FRAME_STEP_SIZE)
    trimmed_trajectories, goals = get_trimmed_trajectories(scenario, episode)

    samples_list = []

    for target_agent_idx, (target_agent_id, trajectory) in enumerate(trimmed_trajectories.items()):

        print('target agent {}/{}'.format(target_agent_idx, len(trimmed_trajectories) - 1))

        reachable_goals_list = get_trajectory_reachable_goals(trajectory, feature_extractor, scenario)
        true_goal_idx = goals[target_agent_id]

        for ego_agent_idx, (ego_agent_id, _) in enumerate(trimmed_trajectories.items()):

            if ego_agent_id == target_agent_id:
                continue

            print('ego agent {}/{}'.format(ego_agent_idx, len(trimmed_trajectories) - 1))

            # If we don't consider occlusions, we don't need the ego vehicle. We thus run the rest of the code once.
            if not missing_feature_extractor and ego_agent_idx != 0:
                break

            if reachable_goals_list[0][true_goal_idx] is not None:

                # get true goal
                true_goal_route = reachable_goals_list[0][true_goal_idx].lane_path
                true_goal_type = feature_extractor.goal_type(true_goal_route)

                if missing_feature_extractor:
                    initial_frame_id_target, last_frame_id_target = get_first_last_frame_ids(episode, target_agent_id)
                    initial_frame_id_ego, last_frame_id_ego = get_first_last_frame_ids(episode, ego_agent_id)
                    initial_frame_id = max(initial_frame_id_target, initial_frame_id_ego)
                    last_frame_id = min(last_frame_id_target, last_frame_id_ego)
                else:
                    initial_frame_id, last_frame_id = get_first_last_frame_ids(episode, target_agent_id)

                for idx, reachable_goals in enumerate(reachable_goals_list):
                    current_frame_id = initial_frame_id + idx

                    if current_frame_id > last_frame_id:
                        break

                    if missing_feature_extractor:
                        occlusions = []

                        frame_occlusions = missing_feature_extractor.occlusions[str(current_frame_id)]
                        for vehicle_occlusions in frame_occlusions:
                            if vehicle_occlusions["ego_agent_id"] == ego_agent_id:
                                occlusions = vehicle_occlusions["occlusions"]

                        target_agent_position = episode.agents[target_agent_id].state.position
                        l = scenario_map.lanes_at(target_agent_position)[0]

                        try:
                            lane_occlusion = occlusions[str(l.parent_road.id)][str(l.id)]
                            lane_occlusion = [Polygon(list(zip(*xy))) for xy in lane_occlusion]

                            if len(lane_occlusion) > 1:
                                lane_occlusion = MultiPolygon(lane_occlusion)
                            else:
                                lane_occlusion = lane_occlusion[0]

                            if lane_occlusion.contains(Point(target_agent_position)):
                                break

                        except KeyError:
                            pass

                    frames = episode_frames[initial_frame_id:current_frame_id + 1]

                    # iterate through each goal for that point in time.
                    for goal_idx, typed_goal in enumerate(reachable_goals):
                        if typed_goal is not None:

                            if missing_feature_extractor:
                                occlusion_features = missing_feature_extractor.extract(target_agent_id,
                                                                                       ego_agent_id,
                                                                                       frames,
                                                                                       typed_goal)
                            features = feature_extractor.extract(target_agent_id, frames, typed_goal)

                            sample = features.copy()
                            sample['agent_id'] = target_agent_id
                            sample['possible_goal'] = goal_idx
                            sample['true_goal'] = true_goal_idx
                            sample['true_goal_type'] = true_goal_type
                            sample['frame_id'] = current_frame_id
                            sample['initial_frame_id'] = initial_frame_id
                            sample['fraction_observed'] = idx / len(reachable_goals_list)

                            # todo: add the occluded features.
                            samples_list.append(sample)

    samples = pd.DataFrame(data=samples_list)
    return samples


def prepare_episode_dataset(params):
    scenario_name, episode_idx = params
    print('scenario {} episode {}'.format(scenario_name, episode_idx))

    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{scenario_name}.xodr")
    scenario_config = ScenarioConfig.load(f"scenarios/configs/{scenario_name}.json")
    scenario = InDScenario(scenario_config)
    feature_extractor = FeatureExtractor(scenario_map)
    missing_feature_extractor = MissingFeatureExtractor(scenario_map, scenario_name, episode_idx)
    episode = scenario.load_episode(episode_idx)


    samples = extract_samples(feature_extractor, scenario, episode, missing_feature_extractor) # todo: always pass
    # todo: missing feature extractor?
    samples.to_csv(get_data_dir() + '{}_e{}.csv'.format(scenario_name, episode_idx), index=False)
    print('finished scenario {} episode {}'.format(scenario_name, episode_idx))
