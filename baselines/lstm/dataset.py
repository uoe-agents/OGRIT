import numpy as np
import pandas as pd
import torch
import pickle

from torch.nn.utils.rnn import pad_sequence
from shapely.geometry import LineString

from igp2.data import ScenarioConfig, InDScenario

from ogrit.core.base import get_scenarios_dir, get_occlusions_dir, get_data_dir
from ogrit.core.feature_extraction import GoalDetector
from ogrit.core.data_processing import get_dataset
from ogrit.occlusion_detection.visualisation_tools import get_box
from ogrit.core.data_processing import FRAME_STEP_SIZE

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
        self.episodes = self.scenario.load_episodes([self.split_type], return_idxes=True)

        self._prepare_dataset()

    def _prepare_dataset(self):
        # todo: get all the possible trimmed trajectories ("sequences"), goals ("labels") and trajectory lengths
        #  ("lengths") that every agent in each of the episodes reached
        trajectories = []
        goals = []
        lengths = []
        for episode_idx, episode in self.episodes:
            trimmed_trajectories, gs, lens = self._trim_trajectories(episode_idx, episode)
            trajectories.extend(trimmed_trajectories)
            goals.extend(gs)
            lengths.extend(lens)
        # todo: make all the trajectories the same length, padding the shortest ones with zeros.
        sequences = pad_sequence(trajectories, batch_first=True, padding_value=0.0)
        goals = torch.LongTensor(goals)
        lengths = torch.Tensor(lengths)
        self.dataset, self.labels, self.lengths = sequences, goals, lengths

    def _trim_trajectories(self, episode_idx, episode):
        trimmed_trajectories = []
        goals = []
        lengths = []

        # detect goal, and trim trajectory past the goal
        goal_detector = GoalDetector(self.scenario.config.goals)

        # todo: load the occlusion dataset for the episode
        with open(get_occlusions_dir() + f"{self.scenario_name}_e{episode_idx}.p", 'rb') as file:
            occlusions = pickle.load(file)

        samples = pd.read_csv(get_data_dir() + f'{self.scenario_name}_e{episode_idx}.csv')

        for agent_id, agent in episode.agents.items():
            if agent.metadata.agent_type in ['car', 'truck_bus']:
                # todo: take the goals that the agent reaches along its path
                agent_goals, goal_frame_idxes = goal_detector.detect_goals(agent.trajectory)

                if len(agent_goals) > 0:
                    # todo: we want the frame in which the agent reached its final goal (the last one in the sequence
                    # todo: is the ultimate agent's goal)

                    ###### TODO:
                    # Get the other non-parked vehicles alive when the agent is.
                    ego_vehicles_id = list(samples[samples["agent_id"] == agent_id]["ego_agent_id"].unique())

                    # Fix the trajectory for the (target) agent to take into account occlusions.
                    for ego_id in ego_vehicles_id:
                        # samples in which both the target and ego are alive
                        relevant_data = samples[(samples["agent_id"] == agent_id) & (samples["ego_agent_id"] == ego_id)]

                        frame_ids = relevant_data["frame_id"].unique()
                        start_frame_id = int(min(frame_ids))
                        end_frame_id = int(max(frame_ids))
                        start_trajectory_idx = start_frame_id - agent.metadata.initial_time

                        # Take trajectories at 1s time steps. todo: improve description
                        for frame_id in range(start_frame_id, end_frame_id, FRAME_STEP_SIZE):
                            end_trajectory_idx = frame_id - agent.metadata.initial_time
                            trimmed_trajectory = agent.trajectory.slice(start_trajectory_idx, end_trajectory_idx)

                            # todo: we want a tensor with all the x,y and heading positions of the agent before it reached
                            #  its goal
                            if isinstance(trimmed_trajectory.path, np.ndarray) and len(trimmed_trajectory.path) > 0:

                                target_trajectory = []

                                for t, _ in enumerate(trimmed_trajectory.timesteps):
                                    x, y = trimmed_trajectory.path[t]
                                    heading = trimmed_trajectory.heading[t]
                                    step_frame_id = start_frame_id + t

                                    target_box = get_box(agent.state, x=x, y=y, heading=heading)
                                    target_box = LineString(target_box.boundary).buffer(0.001)

                                    if target_box.within(occlusions[step_frame_id][ego_id]["occlusions"]):
                                        target_trajectory.append([0, 0, 0, self.OCCLUDED])
                                    else:
                                        target_trajectory.append([x, y, heading, self.NON_OCCLUDED])

                                goals.append(agent_goals[-1])
                                trimmed_trajectories.append(torch.tensor(target_trajectory).float())
                                lengths.append(len(target_trajectory))

        return trimmed_trajectories, goals, lengths


class GRITFeaturesDataset(GRITDataset):
    def __init__(self, scenario_name, split_type="train"):
        super(GRITFeaturesDataset, self).__init__(scenario_name, split_type)

        self._prepare_dataset()

    def _prepare_dataset(self):
        dataset = get_dataset(self.scenario_name, subset=self.split_type)
        dataset = dataset.reset_index(drop=True)

        timing_keys = ["frame_id", "initial_frame_id", "fraction_observed", "episode"]
        # self.timing_info = dataset[["agent_id", "goal_type"] + timing_keys]
        # TODO: check if there are other columns to be dropped after update
        dataset = dataset.drop(timing_keys, axis=1)

        labels = dataset["true_goal"]
        dataset = dataset.drop(["true_goal", "true_goal_type"], axis=1)

        # TODO: convert to int also new boolean columns
        dataset["in_correct_lane"] = dataset["in_correct_lane"].astype(int)
        group_keys = ["agent_id", "possible_goal"]
        groups = dataset.groupby(group_keys).groups
        dataset = pd.get_dummies(dataset, columns=["possible_goal"])
        dataset = dataset.drop(["agent_id", "goal_type"], axis=1)

        sequences = []
        goals = []
        lengths = []
        for key, indices in groups.items():
            if len(indices) > 11 and len(indices) % 11 == 0:
                for i in range(len(indices) // 11):
                    idxs = indices[11 * i:11 * i + 11]
                    sequences.append(torch.Tensor(dataset.loc[idxs].values))
                    most_frequent = labels.loc[idxs].value_counts()
                    goals.append(most_frequent.idxmax())
                    lengths.append(len(idxs))
            elif len(indices) <= 11:
                sequences.append(torch.Tensor(dataset.loc[indices].values))
                most_frequent = labels.loc[indices].value_counts()
                goals.append(most_frequent.idxmax())
                lengths.append(len(indices))

        sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        goals = torch.LongTensor(goals)
        lengths = torch.Tensor(lengths)
        self.dataset, self.labels, self.lengths = sequences, goals, lengths


DATASET_MAP = {"trajectory": GRITTrajectoryDataset, "features": GRITFeaturesDataset}