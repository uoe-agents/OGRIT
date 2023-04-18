from typing import List

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ogrit.core.data_processing import get_multi_scenario_dataset

"""
Create a dataset for the lstm baseline that contains a sequence of features for each agent in each episode.
Since the lstm can be trained and tested on different scenarios, we need to provide a dataset that merges 
the data across scenarios.
"""


class OGRITFeatureDataset(Dataset):

    def __init__(self, scenario_names: List[str], split_type="train"):
        # To convert the goal_type into a hot-one encoding vector. E.g., "continue" -> [0, 0, 1, 0, 0]
        self.le = LabelEncoder()
        self.split_type = split_type

        self.scenario_names = scenario_names

        self.features_to_drop = ["scenario", "episode", "fraction_observed", "initial_frame_id", "frame_id",
                                 "true_goal_type", "true_goal", "goal_type", "ego_agent_id", "agent_id",
                                 "possible_goal", "group_idx"]  # TODO: are there any other feature to drop?

        self.trajectories, self.targets, self.fractions_observed = self.get_samples()

        # self.save() TODO: potentially we could save the dataset to a file to avoid having to recompute it every time

    def get_samples(self):
        """

        Returns:
            trajectories: a list of trajectories, each trajectory is a list of OGRIT features for each timestep
            targets: the true goal type of each trajectory
            (if test dataset) fractions_observed: the fraction of the trajectory that has been observed so far
        """

        samples = get_multi_scenario_dataset(self.scenario_names, self.split_type)

        unique_goal_types = np.unique(samples["true_goal_type"].values)
        self.le.fit(unique_goal_types)

        samples["true_goal_type"] = self.le.transform(samples["true_goal_type"])

        # Group the samples by scenario, episode_id, agent_id and ego_agent_id and possible goal_type.
        # Add index to each group as a column.
        samples["group_idx"] = samples.groupby(["agent_id", "episode", "scenario", "ego_agent_id", "goal_type"],
                                               as_index=False).ngroup()

        trajectories = []
        targets = []
        fractions_observed = []

        for i in range(len(samples)):
            ith_step = samples.iloc[i]

            if self.split_type == "test":
                # The trajectory consists of the steps with the same group_idx as the ith step and with a <= frame_id
                # This will be a dataframe with the different steps as different rows.
                trajectory_steps = samples[(samples["group_idx"] == ith_step["group_idx"])
                                           & (samples["frame_id"] <= ith_step["frame_id"])]

            else:
                # The trajectory consists of the steps with the same group_idx as the ith step.
                trajectory_steps = samples[samples["group_idx"] == ith_step["group_idx"]]

            # The output of this trajectory will be the goal_type of the trajectory after
            # ith_step["fraction_observed"] of the total trajectory executed by the target agent.
            fraction_observed = ith_step["fraction_observed"]

            # Get the fraction observed for each time-step.
            # Drop the features that are not needed.
            trajectory_steps = trajectory_steps.drop(self.features_to_drop, axis=1).values.astype(np.float32)

            # Given the different steps in the trajectory, we want to create a sequence by combining the steps into one
            # input for the lstm. The sequence will be a 1-d array with a tuple of features for each step.
            # E.g., if the trajectory has 2 time-steps, each with 3 features,
            # the sequence will [(f1, f2, f3), (f1, f2, f3)]
            trajectory = torch.tensor([tuple(step) for step in trajectory_steps])

            trajectories.append(trajectory)
            targets.append(ith_step["true_goal_type"])
            fractions_observed.append(fraction_observed)

        trajectories = pad_sequence(trajectories, batch_first=True, padding_value=0.0)
        return trajectories, targets, fractions_observed

    def __getitem__(self, idx):
        """
        Args:
            idx: of the trajectory to return

        Returns:
            trajectory: a sequence of OGRIT features a tensor of shape (seq_len, num_features)
        """

        if self.split_type == "test":
            return self.trajectories[idx], self.targets[idx], self.fractions_observed[idx]
        else:
            return self.trajectories[idx], self.targets[idx]

    def __len__(self):
        """
        Returns:
            the number of trajectories in the dataset
        """

        return len(self.trajectories)

    def get_num_features(self):
        """
        Returns:
            the number of features in each step of the trajectory
        """

        return self.trajectories.shape[2]

    def get_num_classes(self):
        """
        Returns:
            the number of goal_types in the dataset
        """

        return len(self.le.classes_)
