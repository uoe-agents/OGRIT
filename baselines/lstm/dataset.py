import numpy as np
import pandas as pd
import torch
import pickle

from torch.nn.utils.rnn import pad_sequence
from shapely.geometry import LineString

from igp2.data import ScenarioConfig, InDScenario

from ogrit.core.base import get_scenarios_dir, get_occlusions_dir, get_data_dir
from ogrit.core.feature_extraction import GoalDetector
from ogrit.occlusion_detection.visualisation_tools import get_box
from ogrit.core.data_processing import is_target_vehicle_occluded, get_episode_frames

from baselines.lstm.dataset_base import GRITDataset


class GRITTrajectoryDataset(GRITDataset):
    # todo: signal to the LSTM that the vehicle is missing at that timestep
    NON_OCCLUDED = 1
    OCCLUDED = 0

    def __init__(self, scenario_name, split_type="train"):
        super(GRITTrajectoryDataset, self).__init__(scenario_name, split_type)

        scenario_config = ScenarioConfig.load(get_scenarios_dir() + f"configs/{self.scenario_name}.json")
        self.scenario = InDScenario(scenario_config)

        # todo: episodes is a list of
        self.episodes = self.scenario.load_episodes([self.split_type])
        self.episodes = self._get_episodes_idx(self.episodes, scenario_config)

        self._prepare_dataset()

    @staticmethod
    def _get_episodes_idx(episodes, scenario_config):
        """
        Return a list of tuple of the type: (episode_id, episode)
        """
        scenario_episodes = sorted(scenario_config.episodes, key=lambda x: x.recording_id)

        # For each episode in episodes, get their index based on what position they
        return [(i, episode) for episode in episodes for i, s_episode in enumerate(scenario_episodes)
                if episode.config.recording_id == s_episode.recording_id]

    def _prepare_dataset(self):
        # todo: get all the possible trimmed trajectories ("sequences"), goals ("labels") and trajectory lengths
        #  ("lengths") that every agent in each of the episodes reached
        trajectories = []
        goals = []
        lengths = []
        fractions_observed = []
        for episode_idx, episode in self.episodes:
            trimmed_trajectories, gs, lens, fo = self._trim_trajectories(episode_idx, episode)
            trajectories.extend(trimmed_trajectories)
            goals.extend(gs)
            lengths.extend(lens)
            fractions_observed.extend(fo)
        # todo: make all the trajectories the same length, padding the shortest ones with zeros.
        sequences = pad_sequence(trajectories, batch_first=True, padding_value=0.0)
        goals = torch.LongTensor(goals)
        lengths = torch.Tensor(lengths)
        fractions_observed = pad_sequence(fractions_observed, batch_first=True, padding_value=-1)
        self.dataset, self.labels, self.lengths, self.fractions_observed = sequences, goals, lengths, fractions_observed

    def _trim_trajectories(self, episode_idx, episode):
        trimmed_trajectories = []
        goals = []
        lengths = []
        fractions_observed = []

        # detect goal, and trim trajectory past the goal
        goal_detector = GoalDetector(self.scenario.config.goals)
        episode_frames = get_episode_frames(episode)

        # todo: load the occlusion dataset for the episode
        with open(get_occlusions_dir() + f"{self.scenario_name}_e{episode_idx}.p", 'rb') as file:
            occlusions = pickle.load(file)

        samples = pd.read_csv(get_data_dir() + f'{self.scenario_name}_e{episode_idx}.csv')

        for agent_id, agent in episode.agents.items():
            if agent.metadata.agent_type in ['car', 'truck_bus']:

                # todo: take the goals that the agent reaches along its path
                agent_goals, goal_frame_idxes = goal_detector.detect_goals(agent.trajectory)

                if len(agent_goals) < 1:
                    continue

                # Get the target trajectory up until it reaches its possible goal.
                target_trajectory = agent.trajectory.slice(0, goal_frame_idxes[-1]+1)

                # Frame id when the target starts its trajectory.
                target_initial_frame_id = agent.metadata.initial_time

                ###### TODO: add description
                # Get the other non-parked vehicles alive when the agent is.
                ego_vehicles_id = list(samples[samples["agent_id"] == agent_id]["ego_agent_id"].unique())

                # Fix the trajectory for the (target) agent to take into account occlusions.
                for ego_id, ego_agent in episode.agents.items():

                    if ego_id not in ego_vehicles_id:
                        continue

                    # samples in which both the target and ego are alive
                    relevant_data = samples[(samples["agent_id"] == agent_id) & (samples["ego_agent_id"] == ego_id)]

                    frame_ids = relevant_data[["frame_id", "fraction_observed"]].drop_duplicates()

                    # Frame id when the target first became visible to the ego.
                    first_visible_frame_id = min(frame_ids["frame_id"])

                    # Frame id when the target is last visible to the ego, in the SAMPLES. todo
                    last_visible_frame_id = max(frame_ids["frame_id"])

                    # todo: add 1 since we're slicing the trajectory and frame ids start from 0
                    trimmed_trajectory = target_trajectory.slice(first_visible_frame_id - target_initial_frame_id,
                                                                 (last_visible_frame_id+1) - target_initial_frame_id)

                    # todo: we want a tensor with all the x,y and heading positions of the agent up until the
                    #  current frame id
                    target_trajectory_with_occlusions = []

                    if len(trimmed_trajectory.timesteps) == 0:
                        continue

                    for t in range(len(trimmed_trajectory.timesteps)):

                        x, y = trimmed_trajectory.path[t]
                        heading = trimmed_trajectory.heading[t]
                        step_frame_id = first_visible_frame_id + t

                        if is_target_vehicle_occluded(step_frame_id, occlusions, agent_id, ego_id, episode_frames):
                            target_trajectory_with_occlusions.append([0, 0, 0, self.OCCLUDED])
                        else:
                            target_trajectory_with_occlusions.append([x, y, heading, self.NON_OCCLUDED])

                    goals.append(agent_goals[-1])
                    trimmed_trajectories.append(torch.tensor(target_trajectory_with_occlusions).float())
                    lengths.append(len(target_trajectory_with_occlusions))

                    fraction_observed = [(row["frame_id"]-first_visible_frame_id, row["fraction_observed"])
                                         for i, row in frame_ids.iterrows()]
                    fractions_observed.append(torch.Tensor(fraction_observed))

        return trimmed_trajectories, goals, lengths, fractions_observed

DATASET_MAP = {"trajectory": GRITTrajectoryDataset}
