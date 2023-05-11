"""
This file is used to extract the true goal probability data from the pickle file in the `results` folder
which contains the results from the igp2 baselines.
"""

import pickle
import numpy as np
from ogrit.core.base import set_working_dir, get_igp2_results_dir, get_results_dir
import argparse


def create_args():
    config_specification = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    config_specification.add_argument('--scenario', default="rdb5", type=str)

    parsed_config_specification = vars(config_specification.parse_args())
    return parsed_config_specification


if __name__ == '__main__':
    set_working_dir()
    config = create_args()
    scenario_name = config['scenario']
    with open(get_igp2_results_dir() + f'/{scenario_name}_igp2_baseline.pkl', 'rb') as igp2_results:
        igp2_results = pickle.load(igp2_results)[0]

        igp2_probs = igp2_results.true_goal_probability_for_fo

        fraction_observed_grouped = igp2_probs.groupby('fraction_observed')
        true_goal_prob = fraction_observed_grouped.mean()
        true_goal_prob_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())

        true_goal_prob_sem.to_csv(get_results_dir() + f'/{scenario_name}_igp2_true_goal_prob_sem.csv')
        true_goal_prob.to_csv(get_results_dir() + f'/{scenario_name}_igp2_true_goal_prob.csv')
