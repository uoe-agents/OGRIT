import argparse

import numpy as np
import pandas as pd

from model.train_test import FeaturesLSTM
from ogrit.core.base import get_lstm_results_path, get_scenarios_names
from ogrit.core.data_processing import get_multi_scenario_dataset
from ogrit.core.logger import logger

""" 
Train an LSTM baseline and/or evaluate it on the test set.

For example, we can train the model on the heckstrasse and bendplatz scenarios and evaluate it on the 
frankenberg scenario by running:

    python get_results.py --train_scenarios heckstrasse,bendplatz --test_scenarios frankenberg

It saves the probability assigned by the model on the true goal in the /OGRIT/results folder.

Consider plotting the results with the /OGRIT/scripts/plot_results.py script.
"""


# TODO: could add this to evaluate_models.py

def train_lstm(configs):
    lstm = FeaturesLSTM(configs, mode="train")
    lstm.train()


def test_lstm(configs, goal_types):
    goal_probs_df = pd.DataFrame()
    for goal_type in goal_types:
        configs["goal_type"] = goal_type
        lstm = FeaturesLSTM(configs, mode="test")
        goal_probs_df_new = lstm.test()

        # Append the new goal probs to the dataframe goal_probs_df
        goal_probs_df = goal_probs_df.append(goal_probs_df_new, ignore_index=True)

    # normalize the goal probabilities at each time step
    # 1. Take all the samples belonging to the same group (ego-target agent pair) and time frame
    groups = goal_probs_df.groupby(['group_id', 'frame_id'], group_keys=False)

    # 2. Normalize the goal probabilities of different possible reachable goals at each time step
    goal_probs_df['true_goal_prob'] = groups['goal_prob'].apply(lambda x: x / x.sum())

    # 3. Only keep the probabiility associeted with the true goal
    goal_probs_df = goal_probs_df[goal_probs_df['is_true_goal'] == True]

    # save true goal probability
    fraction_observed_grouped = goal_probs_df.groupby('fraction_observed')
    true_goal_prob = fraction_observed_grouped.mean()
    true_goal_prob_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
    goal_prob_file_path, goal_prob_sem_file_path = get_lstm_results_path(
        get_scenarios_names(configs["train_scenarios"]), configs["input_type"],
        get_scenarios_names(configs["test_scenarios"]), configs["update_hz"], configs["fill_occluded_frames_mode"])

    true_goal_prob_sem.to_csv(goal_prob_sem_file_path)
    true_goal_prob.to_csv(goal_prob_file_path)


def get_scenario_goal_types(train_scenarios, update_hz):
    # Load the scenarios data and check the unique values in the sample data

    scenarios_data = get_multi_scenario_dataset(train_scenarios, "train", update_hz=update_hz)
    scenario_goal_types = list(scenarios_data["goal_type"].unique())
    return scenario_goal_types


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and/or evaluate an LSTM baseline with these parameters')

    parser.add_argument('--train_scenarios', type=str, default=None, help='Scenario(s) to use for training. '
                                                                          'Comma-separate the names '
                                                                          'E.g., "heckstrasse,bendplatz"')
    parser.add_argument('--test_scenarios', type=str, required=True, help='Scenario(s) to use for testing. '
                                                                          'Comma-separate the names'
                                                                          'E.g., "heckstrasse,bendplatz"')

    parser.add_argument('--evaluate_only', action='store_true', help='Evaluate an existing model with the '
                                                                     'given hyper-parameters')
    parser.add_argument('--goal_type', type=str, default="all", help="'all', or specific goal type (e.g., 'exit-left')")
    parser.add_argument('--input_type', type=str, default="ogrit_features", help="'absolute_position', "
                                                                                 "'relative_position' or 'ogrit_features'")
    parser.add_argument('--update_hz', type=int, help='take a sample every --update_hz frames in the original episode '
                                                      'frames (e.g., if 25, then take one frame per second)',
                        default=25)
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout for TRAINING. It is 0 for testing.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--lstm_hidden_size', type=int, default=64, help='LSTM hidden dimension')
    parser.add_argument('--fc_hidden_shape', type=int, default=725, help='Fully connected hidden size')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shuffle', action='store_false', help='Shuffle the dataset')
    parser.add_argument('--recompute_dataset', action='store_true',
                        help='Recompute the dataset even if it exists on disk')
    parser.add_argument('--fill_occluded_frames_mode', type=str, default="remove",
                        help='how to fill the frames in the trajectories in which the target is occluded w.r.t the ego. Can be either: '
                             '- "remove" (default): remove the occluded frames'
                             '- "fake_pad": pad the occluded frames with fake values (e.g., -1 for x, y, heading)'
                             '- "use_frame_id": add "frame_id" to the input features (i.e. "tell" the LSTM which frames are occluded)')

    # Parse the arguments into a dictionary
    configs = parser.parse_args()
    configs = vars(configs)

    logger.info(f"Configurations used: {configs}")

    configs["test_scenarios"] = configs["test_scenarios"].split(",")
    configs["train_scenarios"] = configs["train_scenarios"].split(",")

    goal_types_in_train_scenarios = get_scenario_goal_types(configs["train_scenarios"], update_hz=configs["update_hz"])
    goal_types_in_test_scenarios = get_scenario_goal_types(configs["test_scenarios"], update_hz=configs["update_hz"])
    if configs["goal_type"] == "all":
        goal_types = goal_types_in_train_scenarios
    else:
        goal_types = configs["goal_type"].split(",")
        assert all([goal_type in goal_types_in_train_scenarios for goal_type in goal_types]), \
            f"Goal type {goal_type} not available for the training scenarios {configs['train_scenarios']}"

    if not configs["evaluate_only"]:
        for goal_type in goal_types:
            configs["goal_type"] = goal_type

            assert configs["train_scenarios"] is not None, "You must specify the training scenario(s)"

            train_lstm(configs)

    test_lstm(configs, goal_types=goal_types_in_test_scenarios)
