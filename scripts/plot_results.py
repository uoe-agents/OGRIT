import argparse
import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ogrit.core.base import get_base_dir, get_lstm_results_path

parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
parser.add_argument('--models', type=str, help='List of models, comma separated', default='occlusion_grit')
parser.add_argument('--scenarios', type=str, help='List of scenarios, comma separated', default='heckstrasse')
args = parser.parse_args()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.style.use('ggplot')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model_names = args.models.split(',')
scenario_names = args.scenarios.split(',')

# What scenarios we used to train the LSTM. Usually the train and test are the same, but in the generalization
# experiment we train on rdb1,rb2,rb3,rb4,rdb5,rb6,rb7 but test on neuweiler.
lstm_train_scenario = {s: s for s in scenario_names}
lstm_train_scenario["generalization"] = "rdb1_rdb2_rdb3_rdb4_rdb5_rdb6_rdb7"

lstm_test_scenario = {s: s for s in scenario_names}
lstm_test_scenario["generalization"] = "neuweiler"

label_map = {'generalised_grit': 'Oracle',
             'occlusion_grit': 'OGRIT',
             'grit_uniform_prior': 'GRIT',
             'grit': 'GRIT',
             'lstm': 'LSTM',
             'ogrit_oracle': 'OGRIT-oracle',
             'igp2': 'IGP2',
             }

title_map = {'heckstrasse': 'Heckstrasse',
             'bendplatz': 'Bendplatz',
             'frankenburg': 'Frankenburg',
             'neuweiler': 'Neuweiler',
             'rdb1': 'Rdb1',
             'rdb2': 'Rdb2',
             'rdb3': 'Rdb3',
             'rdb4': 'Rdb4',
             'rdb5': 'Rdb5',
             'rdb6': 'Rdb6',
             'rdb7': 'Rdb7',
             'generalization': 'Cross-Dataset', }

plot_accuracy = False
plot_normalised_entropy = False
plot_cross_entropy = False
plot_true_goal_prob = True

results_dir = get_base_dir() + "/results/"

# plot accuracy
if plot_accuracy:
    fig, axes = plt.subplots(2, 2)

    for scenario_idx, scenario_name in enumerate(scenario_names):
        ax = axes[scenario_idx % 2, scenario_idx // 2]
        plt.sca(ax)
        if scenario_idx % 2 == 1:
            plt.xlabel('fraction of trajectory observed')
        plt.title('Accuracy ({})'.format(scenario_name))
        marker = itertools.cycle(('^', '+', 'x', 'o', '*'))

        for model_name in model_names:
            if model_name != 'uniform_prior_baseline':
                accuracy_sem = pd.read_csv(results_dir + f'/{scenario_name}_{model_name}_acc_sem.csv')
                accuracy = pd.read_csv(results_dir + f'/{scenario_name}_{model_name}_acc.csv')

                plt.plot(accuracy.fraction_observed, accuracy.model_correct, label=label_map[model_name],
                         marker=next(marker))
                plt.fill_between(accuracy_sem.fraction_observed, (accuracy + accuracy_sem).model_correct.to_numpy(),
                                 (accuracy - accuracy_sem).model_correct.to_numpy(), alpha=0.2)
        plt.ylim([0, 1])
        plt.legend()

# plot normalised entropy
if plot_normalised_entropy:
    fig, axes = plt.subplots(2, 2)

    for scenario_idx, scenario_name in enumerate(scenario_names):
        ax = axes[scenario_idx % 2, scenario_idx // 2]
        plt.sca(ax)
        if scenario_idx % 2 == 1:
            plt.xlabel('fraction of trajectory observed')
        plt.title('Normalised Entropy ({})'.format(scenario_name))
        marker = itertools.cycle(('^', '+', 'x', 'o', '*'))

        for model_name in model_names:
            entropy_norm_sem = pd.read_csv(results_dir + f'/{scenario_name}_{model_name}_entropy_norm_sem.csv')
            entropy_norm = pd.read_csv(results_dir + f'/{scenario_name}_{model_name}_entropy_norm.csv')

            plt.plot(entropy_norm.fraction_observed, entropy_norm.model_entropy_norm, label=label_map[model_name],
                     marker=next(marker))
            plt.fill_between(entropy_norm_sem.fraction_observed,
                             (entropy_norm + entropy_norm_sem).model_entropy_norm.to_numpy(),
                             (entropy_norm - entropy_norm_sem).model_entropy_norm.to_numpy(), alpha=0.2)
        plt.ylim([0, 1.1])
        plt.legend()

# plot true goal probability
if plot_true_goal_prob:
    plt.rcParams["figure.figsize"] = (15, 15)

    fig, axes = plt.subplots(2, 2)

    for scenario_idx, scenario_name in enumerate(scenario_names):
        ax = axes[scenario_idx % 2, scenario_idx // 2]
        # ax = axes[scenario_idx]
        plt.sca(ax)
        if scenario_idx % 2 == 1:
            plt.xlabel('fraction of trajectory completed')
        if scenario_idx // 2 == 0:
            # if scenario_idx == 0:
            plt.ylabel('Probability assigned to true goal')

        ogrit_marker = None
        ogrit_color = None

        plt.title(title_map[scenario_name])
        marker = itertools.cycle(('^', '+', 'x', 'o', '*'))

        # Plot OGRIT and the baselines.
        for model_name in model_names:

            if model_name == 'occlusion_grit_loocv' and scenario_name == 'neuweiler':
                continue

            if model_name == 'occlusion_grit_rdb5' and scenario_name != 'neuweiler':
                continue

            if "lstm" in model_name:
                _, input_type, update_hz, fill_occluded_frames_mode, features_used_names = model_name.split("-")
                update_hz = int(update_hz)
                try:

                    goal_prob_file_path, goal_prob_sem_file_path = get_lstm_results_path(
                        lstm_train_scenario[scenario_name],
                        input_type,
                        lstm_test_scenario[scenario_name],
                        update_hz,
                        fill_occluded_frames_mode,
                        suffix="",
                        features_used_names=features_used_names, )
                    true_goal_prob_sem = pd.read_csv(goal_prob_sem_file_path)
                    true_goal_prob = pd.read_csv(goal_prob_file_path)
                except FileNotFoundError:
                    continue
            else:
                try:
                    true_goal_prob_sem = pd.read_csv(
                        results_dir + f'/{scenario_name}_{model_name}_true_goal_prob_sem.csv')
                    true_goal_prob = pd.read_csv(results_dir + f'/{scenario_name}_{model_name}_true_goal_prob.csv')
                except FileNotFoundError:
                    continue

            if model_name == 'ogrit_oracle':
                current_marker = None
                color = ogrit_color
                line_style = '--'
            else:
                current_marker = next(marker)
                color = None
                line_style = '-'

            if "lstm" in model_name:
                label = label_map["lstm"]
            else:
                label = label_map[model_name]
            p = plt.plot(np.array(true_goal_prob.fraction_observed), np.array(true_goal_prob.true_goal_prob),
                         line_style,
                         label=label, marker=current_marker, color=color)

            if model_name == 'occlusion_grit':
                ogrit_color = p[0].get_color()
                ogrit_marker = current_marker

            plt.fill_between(true_goal_prob_sem.fraction_observed,
                             (true_goal_prob + true_goal_prob_sem).true_goal_prob.to_numpy(),
                             (true_goal_prob - true_goal_prob_sem).true_goal_prob.to_numpy(), alpha=0.2,
                             color=p[0].get_color())
        plt.ylim([0.0, 1.0])
        if scenario_idx == 0:
            # plot legend in small font
            plt.legend(fontsize=8)
    plt.savefig(get_base_dir() + '/images/true_goal_prob_ogrit.pdf', bbox_inches='tight')

# plot cross entropy
if plot_cross_entropy:
    fig, axes = plt.subplots(2, 2)

    for scenario_idx, scenario_name in enumerate(scenario_names):
        ax = axes[scenario_idx % 2, scenario_idx // 2]
        plt.sca(ax)
        if scenario_idx % 2 == 1:
            plt.xlabel('fraction of trajectory observed')
        plt.title('Cross Entropy ({})'.format(scenario_name))
        marker = itertools.cycle(('^', '+', 'x', 'o', '*'))

        for model_name in model_names:

            if model_name == 'lstm':
                continue

            cross_entropy_sem = pd.read_csv(results_dir + f'/{scenario_name}_{model_name}_cross_entropy_sem.csv')
            cross_entropy = pd.read_csv(results_dir + f'/{scenario_name}_{model_name}_cross_entropy.csv')

            plt.plot(cross_entropy.fraction_observed, cross_entropy.cross_entropy, label=label_map[model_name],
                     marker=next(marker))
            plt.fill_between(cross_entropy.fraction_observed,
                             (cross_entropy + cross_entropy_sem).cross_entropy.to_numpy(),
                             (cross_entropy - cross_entropy_sem).cross_entropy.to_numpy(), alpha=0.2)
        plt.ylim([0, 1.1])
        plt.legend()

plt.show()
