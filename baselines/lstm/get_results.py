import argparse

from model.train_test import FeaturesLSTM

""" 
Train an LSTM baseline and/or evaluate it on the test set.

For example, we can train the model on the heckstrasse and bendplatz scenarios and evaluate it on the 
frankenberg scenario by running:

    python get_results.py --train_scenarios heckstrasse,bendplatz --test_scenarios frankenberg

It saves the probability assigned by the model on the true goal in the /OGRIT/results folder.

Consider plotting the results with the /OGRIT/scripts/plot_results.py script.
"""

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
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout for TRAINING. It is 0 for testing.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--lstm_hidden_size', type=int, default=64, help='LSTM hidden dimension')
    parser.add_argument('--fc_hidden_shape', type=int, default=725, help='Fully connected hidden size')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shuffle', action='store_false', help='Shuffle the dataset')

    # Parse the arguments into a dictionary
    configs = parser.parse_args()
    configs = vars(configs)


    def train_lstm():
        lstm = FeaturesLSTM(configs, mode="train")
        lstm.train()


    def test_lstm():
        lstm = FeaturesLSTM(configs, mode="test")
        lstm.test()


    if configs["evaluate_only"]:
        test_lstm()
    else:
        assert configs["train_scenarios"] is not None, "You must specify the training scenario(s)"

        train_lstm()
        test_lstm()
