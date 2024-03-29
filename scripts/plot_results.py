import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ogrit.core.base import get_base_dir, get_all_scenarios
import baselines.lstm.test as eval_lstm
import itertools
import argparse
import os

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.style.use('ggplot')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

scenario_names = get_all_scenarios()
model_names = ['occlusion_grit', 'ogrit_oracle', 'grit_no_missing_uniform', 'lstm', 'igp2', 'occlusion_grit_rdb5', 'uniform_prior_baseline']

label_map = {'generalised_grit': 'Oracle',
             'occlusion_grit': 'OGRIT',
             'occlusion_grit_loocv': 'OGRIT-LOOCV',
             'occlusion_baseline': 'truncated G-GRIT',
             'no_possibly_missing_features_ogrit': 'OGRIT baseline',
             'uniform_prior_baseline': 'OGRIT-no-DT',
             'grit_uniform_prior': 'GRIT',
             'grit': 'GRIT',
             'lstm': 'LSTM',
             'sogrit': 'S-OGRIT',
             'ogrit_oracle': 'OGRIT-oracle',
             'trained_trees': 'GRIT',
             'truncated_grit': 'Truncated GRIT',
             'no_possibly_missing_features_grit': 'GRIT',
             'grit_no_missing_uniform': 'GRIT',
             'igp2': 'IGP2',
             'occlusion_grit_rdb5': 'OGRIT rdb5'}

title_map = {'heckstrasse': 'Heckstrasse',
             'bendplatz': 'Bendplatz',
             'frankenburg': 'Frankenburg',
             'neuweiler': 'Neuweiler',
             'neukoellnerstrasse': 'Neukoellner Strasse',
             'rdb5': 'Rdb5'}

plot_accuracy = False
plot_normalised_entropy = False
plot_cross_entropy = False
plot_true_goal_prob = True

results_dir = get_base_dir() + "/results/"
#results_dir = get_base_dir() + f'/results/loocv/'

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

                plt.plot(accuracy.fraction_observed, accuracy.model_correct, label=label_map[model_name], marker=next(marker))
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

            plt.plot(entropy_norm.fraction_observed, entropy_norm.model_entropy_norm, label=label_map[model_name], marker=next(marker))
            plt.fill_between(entropy_norm_sem.fraction_observed, (entropy_norm + entropy_norm_sem).model_entropy_norm.to_numpy(),
                             (entropy_norm - entropy_norm_sem).model_entropy_norm.to_numpy(), alpha=0.2)
        plt.ylim([0, 1.1])
        plt.legend()


# plot true goal probability
if plot_true_goal_prob:
    plt.rcParams["figure.figsize"] = (16, 3)

    fig, axes = plt.subplots(1, 4)


    def plot_lstm(scenario_name, label, marker):
        lstm_dataset = "trajectory"

        # Plot LSTM
        test_config = argparse.Namespace(**{
            "dataset": lstm_dataset,
            "shuffle": True,
            "scenario": scenario_name,
            "model_path": f"/checkpoint/{scenario_name}_{lstm_dataset}_best.pt",
            "lstm_hidden_dim": 64,
            "fc_hidden_dim": 725,
            "lstm_layers": 1,
            "step": 0.1
        })
        lstm_probs, _ = eval_lstm.main(test_config)

        fraction_observed_grouped = lstm_probs.groupby('fraction_observed')
        true_goal_prob = fraction_observed_grouped.mean()
        true_goal_prob_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())

        xs = np.arange(fraction_observed_grouped.ngroups)
        plt.plot(xs, true_goal_prob.true_goal_prob, label=label,
                              marker=marker)
        plt.fill_between(xs,
                         (true_goal_prob + true_goal_prob_sem).true_goal_prob.to_numpy(),
                         (true_goal_prob - true_goal_prob_sem).true_goal_prob.to_numpy(), alpha=0.2)


    for scenario_idx, scenario_name in enumerate(scenario_names):
        #ax = axes[scenario_idx % 2, scenario_idx // 2]
        ax = axes[scenario_idx]
        plt.sca(ax)
        # if scenario_idx % 2 == 1:

        plt.xlabel('fraction of trajectory completed')
        # if scenario_idx // 2 == 0:
        if scenario_idx == 0:
            plt.ylabel('Probability assigned to true goal')

        ogrit_marker = None
        ogrit_color = None


        plt.title(title_map[scenario_name])
        marker = itertools.cycle(('^', '+', 'x', 'o', '*'))

        # Plot OGRIT and the baselines.
        for model_name in model_names:

            # if model_name == "lstm":
            #     plot_lstm(scenario_name, label=label_map[model_name], marker=next(marker))
            #     continue

            if model_name == 'occlusion_grit_loocv' and scenario_name == 'neuweiler':
                continue

            if model_name == 'occlusion_grit_rdb5' and scenario_name != 'neuweiler':
                continue

            true_goal_prob_sem = pd.read_csv(results_dir + f'/{scenario_name}_{model_name}_true_goal_prob_sem.csv')
            true_goal_prob = pd.read_csv(results_dir + f'/{scenario_name}_{model_name}_true_goal_prob.csv')

            if model_name == 'ogrit_oracle':
                current_marker = None
                color = ogrit_color
                line_style = '--'
            else:
                current_marker = next(marker)
                color = None
                line_style = '-'

            p = plt.plot(true_goal_prob.fraction_observed, true_goal_prob.true_goal_prob, line_style,
                         label=label_map[model_name], marker=current_marker, color=color)

            if model_name == 'occlusion_grit':
                ogrit_color = p[0].get_color()
                ogrit_marker = current_marker

            plt.fill_between(true_goal_prob_sem.fraction_observed, (true_goal_prob + true_goal_prob_sem).true_goal_prob.to_numpy(),
                             (true_goal_prob - true_goal_prob_sem).true_goal_prob.to_numpy(), alpha=0.2, color=p[0].get_color())
        plt.ylim([0.0, 1.0])
        if scenario_idx == 0:
            plt.legend()
        plt.legend()
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

            plt.plot(cross_entropy.fraction_observed, cross_entropy.cross_entropy, label=label_map[model_name], marker=next(marker))
            plt.fill_between(cross_entropy.fraction_observed, (cross_entropy + cross_entropy_sem).cross_entropy.to_numpy(),
                             (cross_entropy - cross_entropy_sem).cross_entropy.to_numpy(), alpha=0.2)
        plt.ylim([0, 1.1])
        plt.legend()

plt.show()
