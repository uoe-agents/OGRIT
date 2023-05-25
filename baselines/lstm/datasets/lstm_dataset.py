import os
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from ogrit.core.base import get_lstm_dir
from ogrit.core.data_processing import get_multi_scenario_dataset
from ogrit.core.feature_extraction import FeatureExtractor
from ogrit.core.logger import logger

"""
Define the skeleton of the base class that each other dataset should inherit # TODO: what is this called?
To define a new Dataset, create a new class that inherits from this one and change the get_samples() function 
based on what information the input trajectories should have (e.g., OGRIT features such as speed, in_correct_lane, ...) 
or position (x, y, heading at each timestep), etc.
"""


class LSTMDataset(Dataset):

    def __init__(self, scenario_names: List[str], input_type, split_type, update_hz):
        """
        Return the trajectories we need to pass the LSTM for given scenarios and features.

        Args:
            scenario_names: Name of the scenarios for which we want the data
            input_type: what features do we want to keep from: "absolute_position", "relative_position", "ogrit_features".
                    'absolute_position' = the trajectories are a sequence of (x,y,heading) steps
                    'relative_position' =  "     "           "  "   "      "  path_to_goal_length
                    'ogrit_features' =     "     "           "  "   "      "  (all the features used by OGRIT) steps
            split_type: "train" or "test"
            update_hz: take a sample every update_hz frames in the original episode frames (e.g., if 25, then take
                        one frame per second)
        """

        # To convert the goal_type into a hot-one encoding vector. E.g., "continue" -> [0, 0, 1, 0, 0]
        self.le = LabelEncoder()
        self.split_type = split_type
        self.input_type = input_type
        self.update_hz = update_hz

        self.scenario_names = scenario_names

        if input_type == 'absolute_position':
            self.features_to_use = ['x', 'y', 'heading']
        elif input_type == 'relative_position':
            self.features_to_use = ['delta_x_from_possible_goal', 'delta_y_from_possible_goal']
        elif input_type == 'ogrit_features':
            self.features_to_use = FeatureExtractor.feature_names.keys()
        else:
            raise ValueError(f"Parameter type should be either 'absolute_position', 'relative_position', "
                             f"'ogrit_features', but got {input_type} instead.")

        self._trajectories, self._targets, self._fractions_observed = self.load_dataset()

    def load_dataset(self):

        dataset_path = get_lstm_dir() + f"/datasets/{'_'.join(self.scenario_names)}_{self.input_type}_{self.split_type}.pt"
        if not os.path.exists(dataset_path):
            logger.info(f"Creating dataset {dataset_path}...")
            trajectories, targets, fractions_observed = self.get_dataset()
            torch.save({"dataset": trajectories,
                        "targets": targets,
                        # "lengths": dataset.lengths, TODO: use lengths?
                        "fractions_observed": fractions_observed},
                       dataset_path)
        else:
            logger.info(f"Loading dataset {dataset_path}...")
            dataset_dict = torch.load(dataset_path)
            trajectories = dataset_dict["dataset"]
            targets = dataset_dict["targets"]
            # dataset.lengths = dataset_dict["lengths"] TODO: use lengths?
            fractions_observed = dataset_dict["fractions_observed"]
        return trajectories, targets, fractions_observed

    def get_dataset(self):
        """
        This function is what changes across datasets.

        Returns:
            trajectories: a list of trajectories. Each trajectory contains information (depending on the dataset
                          we want) for each timestep. For example, if we want to give the lstm the x, y, heading
                          position, then we should output np.torch([[[x_10, y_10, h_10], ..., [x_1n, y_1n, h_1n]],
                          [[x_k0, y_k0, h_k0], ..., [x_kn, y_kn, h_kn]], ...]) for k trajectories, each of n steps
            targets: the true goal type of each trajectory
            (if test dataset) fractions_observed: the fraction of the trajectory that has been observed so far
        """
        samples = self.get_samples()

        trajectories = []
        targets = []
        fractions_observed = []

        for i in tqdm(range(len(samples))):
            ith_step = samples.iloc[i]

            # TODO: add explanation for why this way -- maybe use a mask of embedding instead?
            if self.split_type == "test":
                # The trajectory consists of the steps with the same group_idx as the ith step and with a <= frame_id
                # This will be a dataframe with the different steps as different rows.
                trajectory_steps = samples[(samples["group_idx"] == ith_step["group_idx"])
                                           & (samples["frame_id"] <= ith_step["frame_id"])]

            else:
                # The trajectory consists of the steps with the same group_idx as the ith step.
                # TODO: there are multiple lines in the df, one for each possible goal
                trajectory_steps = samples[samples["group_idx"] == ith_step["group_idx"]]

            # The output of this trajectory will be the goal_type of the trajectory after
            # ith_step["fraction_observed"] of the total trajectory executed by the target agent.
            fraction_observed = ith_step["fraction_observed"]

            # Get the data in the samples for the features we want.
            trajectory_steps = trajectory_steps[self.features_to_use].values.astype(np.float32)

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

    def get_samples(self):
        """
        Returns: the samples used by OGRIT for the scenarios required. Change the "true_goal_type" column to be
        a scalar rather than a string (e.g., "straight-on" becomes 1, etc), so that we can use it with an LSTM.


        """

        samples = get_multi_scenario_dataset(self.scenario_names, self.split_type, update_hz=self.update_hz)

        unique_goal_types = np.unique(samples["true_goal_type"].values)
        self.le.fit(unique_goal_types)

        samples["true_goal_type"] = self.le.transform(samples["true_goal_type"])

        # Group the samples by scenario, episode_id, agent_id and ego_agent_id and possible goal_type.
        # Add index to each group as a column.
        samples["group_idx"] = samples.groupby(["agent_id", "episode", "scenario", "ego_agent_id", "goal_type"],
                                               as_index=False).ngroup()

        return samples

    def __getitem__(self, idx):
        """
        Args:
            idx: of the trajectory to return

        Returns:
            trajectory: a sequence of OGRIT features a tensor of shape (seq_len, num_features)
        """

        if self.split_type == "test":
            return self._trajectories[idx], self._targets[idx], self._fractions_observed[idx]
        else:
            return self._trajectories[idx], self._targets[idx]

    def __len__(self):
        """
        Returns:
            the number of trajectories in the dataset
        """

        return len(self._trajectories)

    def get_num_features(self):
        """
        Returns:
            the number of features in each step of the trajectory
        """

        return self._trajectories.shape[2]

    def get_num_classes(self):
        """
        Returns:
            the number of goal_types in the dataset
        """

        return len(np.unique(self._targets))
