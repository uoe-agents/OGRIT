import argparse
import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ogrit.core.base import get_base_dir


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.style.use('ggplot')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

lstm_train_scenario = {"heckstrasse": "heckstrasse",
                       "bendplatz": "bendplatz",
                       "frankenburg": "frankenburg",
                       "rdb3": "rdb3"}


label_map = {'occlusion_grit_variant_1': 'OGRIT generalisation',
             'occlusion_grit_variant_2': 'OGRIT generalisation',
             'occlusion_grit_variant_3': 'OGRIT generalisation',
             'occlusion_grit_variant_1_new_features': 'OGRIT generalisation',
             'occlusion_grit_variant_2_new_features': 'OGRIT generalisation',
             'occlusion_grit_variant_3_new_features': 'OGRIT generalisation',
             'occlusion_grit_all_roundabouts': 'OGRIT trained on all',
             'occlusion_grit_all_roundabouts_new_features': 'OGRIT trained on all',
             'igp2': 'IGP2',
             'occlusion_grit_all_opendd': 'OGRIT openDD',
             'occlusion_grit_all_opendd_new_features': 'OGRIT openDD'}


experiments = {'Variant 1': {'models': ['occlusion_grit_variant_1_new_features', 'occlusion_grit_all_roundabouts_new_features'],
                             'test': ['neuweiler', 'rdb4', 'rdb5']},
               'Variant 2': {'models': ['occlusion_grit_variant_2_new_features', 'occlusion_grit_all_roundabouts_new_features'],
                             'test': ['neuweiler', 'rdb5']},
               'Variant 3': {'models': ['occlusion_grit_variant_3_new_features', 'occlusion_grit_all_roundabouts_new_features'],
                             'test': ['neuweiler']},
               'Cross-Dataset': {'models': ['occlusion_grit_all_opendd_new_features', 'occlusion_grit_all_roundabouts_new_features'],
                                 'test': ['neuweiler']}}


results_dir = get_base_dir() + "/results/"

# plot true goal probability
plt.rcParams["figure.figsize"] = (10, 10)

fig, axes = plt.subplots(2, 2)

for experiment_idx, experiment_name in enumerate(experiments):
    ax = axes[experiment_idx % 2, experiment_idx // 2]
    # ax = axes[scenario_idx]
    plt.sca(ax)
    if experiment_idx % 2 == 1:
        plt.xlabel('fraction of trajectory completed')
    if experiment_idx // 2 == 0:
        # if scenario_idx == 0:
        plt.ylabel('Probability assigned to true goal')

    ogrit_marker = None
    ogrit_color = None

    plt.title(experiment_name)
    marker = itertools.cycle(('^', '+', 'x', 'o', '*'))

    # Plot OGRIT and the baselines.
    for model_name in experiments[experiment_name]['models']:
        true_goal_probs = []
        true_goal_prob_sems = []
        fraction_observed = np.linspace(0, 1, 11)
        for scenario_name in experiments[experiment_name]['test']:
            if "lstm" in model_name:
                try:
                    true_goal_prob_sem = pd.read_csv(
                        results_dir + f'/{scenario_name}_{model_name}_on_{lstm_train_scenario[scenario_name]}_true_goal_prob_sem.csv')
                    true_goal_prob = pd.read_csv(
                        results_dir + f'/{scenario_name}_{model_name}_on_{lstm_train_scenario[scenario_name]}_true_goal_prob.csv')
                except FileNotFoundError:
                    continue
            else:
                try:
                    true_goal_prob_sem = pd.read_csv(
                        results_dir + f'/{scenario_name}_{model_name}_true_goal_prob_sem.csv')
                    true_goal_prob = pd.read_csv(results_dir + f'/{scenario_name}_{model_name}_true_goal_prob.csv')
                except FileNotFoundError:
                    continue
            true_goal_probs.append(np.array(true_goal_prob.true_goal_prob))
            true_goal_prob_sems.append(np.array(true_goal_prob_sem.true_goal_prob))

        current_marker = next(marker)
        color = None
        line_style = '-'

        true_goal_prob = np.stack(true_goal_probs).mean(axis=0)
        true_goal_prob_sem = np.stack(true_goal_prob_sems).mean(axis=0) / np.sqrt(len(true_goal_prob_sems))

        label = label_map[model_name] if model_name in label_map else model_name

        p = plt.plot(fraction_observed, true_goal_prob, line_style,
                     label=label, marker=current_marker, color=color)

        plt.fill_between(fraction_observed,
                         true_goal_prob + true_goal_prob_sem,
                         true_goal_prob - true_goal_prob_sem, alpha=0.2,
                         color=p[0].get_color())
    plt.ylim([0.0, 1.0])
    if experiment_idx == 0:
        plt.legend()

plt.savefig(get_base_dir() + '/images/ogrit_generalisation.pdf', bbox_inches='tight')

plt.show()
