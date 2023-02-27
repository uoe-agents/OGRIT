import argparse
import json
import time

import pandas as pd
import numpy as np
import torch
from ogrit.core.base import get_lstm_dir, get_results_dir
from torch.utils.data import DataLoader

from baselines.lstm.model import LSTMModel
from baselines.lstm.train import load_save_dataset, logger



def main(config):
    torch.random.manual_seed(42)

    if hasattr(config, "config"):
        config = argparse.Namespace(**json.load(open(config.config)))
    logger.info(config)

    scenario_name = config.scenario
    test_dataset = load_save_dataset(config, "test")
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=len(test_dataset))
    test_data = [_ for _ in test_loader][0]
    logger.info(f"Running testing")

    model_dict = torch.load(get_lstm_dir() + config.model_path, map_location=torch.device('cpu'))
    model = LSTMModel(test_dataset.dataset.shape[-1],
                      config.lstm_hidden_dim,
                      config.fc_hidden_dim,
                      test_dataset.labels.unique().shape[-1],
                      num_layers=config.lstm_layers,
                      dropout=0.0)

    try:
        model.load_state_dict(model_dict["model_state_dict"])
    except RuntimeError:
        # We used DataParallel during training todo: could save dict withoutd "model" using torch.save(model.module.state_dict(), "model_ckpt.py")
        pretrained_dict = model_dict["model_state_dict"]
        pretrained_dict = {key.replace("module.", ""): value for key, value, in pretrained_dict.items()}
        model.load_state_dict(pretrained_dict)

    start = time.time()

    trajectories = test_data[0]
    target = test_data[1]
    fractions_observed = test_data[3].tolist()

    model.eval()

    output, (encoding, lengths) = model(trajectories, use_encoding=True, device="cpu")

    goal_probs = torch.exp(encoding).detach().numpy()

    # For each trajectory, take the points at every "step" distance in the path.
    goal_probs_df = {"true_goal_prob": [], "fraction_observed": []}

    # For each trajectory, take the steps in which we have data in te OGRIT dataset todo
    for trajectory_idx in range(len(fractions_observed)):

        fractions_for_trajectory = fractions_observed[trajectory_idx]

        for fraction in fractions_for_trajectory:

            frame, fo = fraction
            frame = int(frame)

            if frame == -1:
                # We reached the end of the frames (as -1 is the padding value)
                break

            fo = round(fo, 1)

            true_goal = target[trajectory_idx]
            true_goal_prob = goal_probs[trajectory_idx][frame][true_goal]
            goal_probs_df["true_goal_prob"].append(true_goal_prob)
            goal_probs_df["fraction_observed"].append(fo)

    dur = time.time() - start
    goal_probs_df = pd.DataFrame(goal_probs_df)

    # save true goal probability
    fraction_observed_grouped = goal_probs_df.groupby('fraction_observed')
    true_goal_prob = fraction_observed_grouped.mean()
    true_goal_prob_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())

    true_goal_prob_sem.to_csv(get_results_dir() + f'/{scenario_name}_lstm_true_goal_prob_sem.csv')
    true_goal_prob.to_csv(get_results_dir() + f'/{scenario_name}_lstm_true_goal_prob.csv')

    return goal_probs_df, dur


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