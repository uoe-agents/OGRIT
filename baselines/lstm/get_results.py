import argparse

from model.train_test import FeaturesLSTM
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


def test_lstm(configs):
    lstm = FeaturesLSTM(configs, mode="test")
    lstm.test()


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

    # Parse the arguments into a dictionary
    configs = parser.parse_args()
    configs = vars(configs)

    logger.info(f"Configurations used: {configs}")

    if configs["evaluate_only"]:
        test_lstm(configs)
    else:
        assert configs["train_scenarios"] is not None, "You must specify the training scenario(s)"

        train_lstm(configs)
        test_lstm(configs)
