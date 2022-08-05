import argparse
import json
import time

import numpy as np
import torch
from ogrit.core.base import get_lstm_dir
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from baselines.lstm.model import LSTMModel
from baselines.lstm.train import load_save_dataset, logger


def main(config):
    torch.random.manual_seed(42)

    if hasattr(config, "config"):
        config = argparse.Namespace(**json.load(open(config.config)))
    logger.info(config)

    test_dataset = load_save_dataset(config, "test")
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=len(test_dataset))
    test_data = [_ for _ in test_loader][0]
    logger.info(f"Running testing")

    model_dict = torch.load(get_lstm_dir() + config.model_path)
    model = LSTMModel(test_dataset.dataset.shape[-1],
                      config.lstm_hidden_dim,
                      config.fc_hidden_dim,
                      test_dataset.labels.unique().shape[-1],
                      num_layers=config.lstm_layers,
                      dropout=0.0)
    model.load_state_dict(model_dict["model_state_dict"])

    start = time.time()

    trajectories = test_data[0]
    target = test_data[1]
    lengths = test_data[2]
    fractions_observed = test_data[3].tolist()

    input = pack_padded_sequence(trajectories, lengths, batch_first=True, enforce_sorted=False)

    model.eval()

    output, (encoding, lengths) = model(input, use_encoding=True)

    matches = (encoding.argmax(axis=-1) == target.unsqueeze(-1)).to(float)
    mask = (torch.arange(encoding.shape[1])[None, :] >= lengths[:, None])
    matches = matches.masked_fill(mask, 0)
    goal_probs = torch.exp(encoding)

    step = 0.1  # todo: config.step
    count = int(1 / step + 1)
    if encoding.shape[1] > count:  # todo, necessary given below it's determinsitic and not dynamic?
        # For each trajectory, take the points at every "step" distance in the path. todo: is it lenghts[i] + 1?

        corrects = {round(k, 1): [] for k in np.linspace(0, 1, count)}  # todo: explain
        goal_probs_grouped = {round(k, 1): [] for k in np.linspace(0, 1, count)}

        for i in range(len(fractions_observed)):
            fo = round(fractions_observed[i], 1)
            corrects[fo].append(matches[i][lengths[i] - 1])
            goal_probs_grouped[fo].append(goal_probs[i][lengths[i] - 1])

        # take the prediction at the length-step and give the fraction observed as x axis
    else:
        # todo: update itttt
        corrects = (encoding.argmax(axis=-1) == target.unsqueeze(-1)).to(float)

    dur = time.time() - start

    return corrects, goal_probs, dur

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Location of config file ending with *.json. Specifying this will"
                                                   "overwrite all other arguments.")
    parser.add_argument("--scenario", type=str, help="Scenario to train on")
    parser.add_argument("--dataset", type=str, help="Whether to use trajectory or features dataset")
    parser.add_argument("--shuffle", "-s", action="store_true", help="Shuffle dataset before sampling")
    parser.add_argument("--save_path", type=str, help="Save path for model checkpoints.")

    args = parser.parse_args()

    main(args)