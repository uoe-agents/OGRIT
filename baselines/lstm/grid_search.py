import itertools
import argparse
import os
import numpy as np
import logging
import copy
import torch
import pandas as pd
import sys
from baselines.lstm.train import train, logger

grid_search_params = {
    "lstm_hidden_dim": np.logspace(4, 13, 5, base=2, dtype=int),
    "fc_hidden_dim": np.linspace(100, 1000, 5, dtype=int),
    "lstm_layers": [1, 3, 5],
    "lr": [0.01, 0.005, 0.001],
    "dropout": np.arange(0.1, 1.0, 0.25)
}


def setup_logging(logger):
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

# todo: grid search to find the best hyper-parameters
if __name__ == '__main__':
    type = sys.argv[1]  # search/best
    dataset = sys.argv[2]  # features/trajectory
    scenario = sys.argv[3]  # heckstrasse/bendplatz/frankenberg/round
    i = int(sys.argv[4])

    path_string = "grid_search/{0}_{1}_{2}_{3}_{4}_{5}_{6:.2f}"

    if type == "search":
        if not os.path.exists("grid_search"):
            os.mkdir("grid_search")

        grid_params = list(product_dict(**grid_search_params))
        params = grid_params[i]
        save_path = path_string.format(
            scenario, dataset,
            params['lstm_hidden_dim'],
            params["fc_hidden_dim"],
            params["lstm_layers"],
            params['lr'],
            params['dropout'])
        params.update({"dataset": dataset, "shuffle": True, "batch_size": 10, "max_epoch": 100, "scenario": scenario,
                       "save_path": save_path, "save_latest": False, "use_encoding": True})
        params = argparse.Namespace(**params)
        train(params)

    elif type == "best":
        results = []
        for params in product_dict(**grid_search_params):
            save_path = path_string.format(
                scenario, dataset,
                params['lstm_hidden_dim'],
                params["fc_hidden_dim"],
                params["lstm_layers"],
                params['lr'],
                params['dropout']) + f"_{scenario}_{dataset}"
            try:
                best = torch.load(save_path + "_best.pt")
            except IOError as e:
                logger.exception(str(e), exc_info=e)
                continue
            result = copy.copy(params)
            result.update({"best_loss": best["losses"].min(), "best_acc": best["accs"].max(),
                           "best_epoch": best["epoch"]},
                          )
            results.append(result)
        results = pd.DataFrame(results)
        results.to_csv("grid_search/results.csv")

        logger.info(f"The best parameter by loss:\n{results.loc[results.idxmin()['best_loss']]}")
