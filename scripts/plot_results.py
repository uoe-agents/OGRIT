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
parser.add_argument('--lstm_train_scenario', type=str,
                    help='Use the LSTM model trained on this scenario(s) to evaluate the test --scenarios. Own = use the same scenario as the testing one',
                    default='own')
args = parser.parse_args()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.style.use('ggplot')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# which model(s) we used to train the scenario(s) we are evaluating
model_names = args.models.split(',')
scenario_names = args.scenarios.split(',')

if args.lstm_train_scenario == "own":
    lstm_train_scenario = {s: s for s in scenario_names}
elif args.lstm_train_scenario == "variants":
    lstm_train_scenario = {"variant1": 'rdb2_rdb3_rdb6_rdb7',
                           "variant2": 'rdb2_rdb3_rdb4_rdb6_rdb7',
                           "variant3": 'rdb2_rdb3_rdb4_rdb5_rdb6_rdb7',
                           "variant4": 'rdb1_rdb2_rdb3_rdb4_rdb5_rdb6_rdb7'}
else:
    lstm_train_scenario = {s: args.lstm_train_scenario for s in scenario_names}

label_map = {'generalised_grit': 'Oracle',
             'occlusion_grit': 'OGRIT',
             'occlusion_grit_loocv': 'OGRIT-LOOCV',
             'occlusion_baseline': 'truncated G-GRIT',
             'no_possibly_missing_features_ogrit': 'OGRIT baseline',
             'uniform_prior_baseline': 'OGRIT-no-DT',
             'grit_uniform_prior': 'GRIT',
             'grit': 'GRIT',
             'lstm': 'LSTM',
             'lstm_ogrit_features_all': 'LSTM OGRIT features',
             'lstm_relative_position_all': 'LSTM relative position',
             'lstm_absolute_position': 'LSTM abs',
             'sogrit': 'S-OGRIT',
             'ogrit_oracle': 'OGRIT-oracle',
             'trained_trees': 'GRIT',
             'truncated_grit': 'Truncated GRIT',
             'no_possibly_missing_features_grit': 'GRIT',
             'grit_no_missing_uniform': 'GRIT',
             'igp2': 'IGP2',
             'occlusion_grit_rdb5': 'OGRIT rdb5',
             'occlusion_grit_old_features': 'old features',
             'occlusion_grit_new_features': 'new features',
             'occlusion_gritall_rdbs_angle_to_goal': 'ogrti w/ angle_to_goal',
             'occlusion_grit_all_rdbs_angle_to_goal_remove_goal_passed': 'ogrit w/ angle_to_goal + remove last passed goal',
             'occlusion_grit_all_rdbs_angle_to_goal_angle_ch123s': 'angle_change1/2/3s',
             'occlusion_grit_all_rdbs_abs_angle_to_goal_remove_goal_passed': 'abs_angle_to_goal',
             'occlusion_grit_all_rdbs_negative_abs_angle_to_goal_remove_goal_passed': 'neg_abs_angle_to_goal',
             'lstm_ogrit_features_occluded_features_no_angle_to_goal': 'lstm no_angle_to_goal',
             'lstm_ogrit_features_angle_to_goal_in_correct_lane': 'lstm only angle_to_goal + correct lane',
             'lstm_ogrit_features_speed_in_correct_lane': 'lstm only speed + correct lane',
             'lstm_ogrit_features_path_to_goal_length_in_correct_lane': 'lstm only path_to_goal + correct lane',
             'lstm_ogrit_features_angle_in_lane_in_correct_lane': 'lstm only angle in lane + correct lane',
             'lstm_ogrit_features_old': 'old LSTM w/o angle to goal but no occluded features'
             }

title_map = {'heckstrasse': 'Heckstrasse',
             'bendplatz': 'Bendplatz',
             'frankenburg': 'Frankenburg',
             'neuweiler': 'Neuweiler',
             'neukoellnerstrasse': 'Neukoellner Strasse',
             'rdb1': 'Rdb1',
             'rdb2': 'Rdb2',
             'rdb3': 'Rdb3',
             'rdb4': 'Rdb4',
             'rdb5': 'Rdb5',
             'rdb6': 'Rdb6',
             'rdb7': 'Rdb7', }

if args.lstm_train_scenario == "variants":
    title_map = {"variant1": 'Variant 1',
                 "variant2": 'Variant 2',
                 "variant3": 'Variant 3',
                 "variant4": 'Variant 4'}

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

                    test_scenario_variants = {"variant1": "neuweiler_rdb4_rdb5",
                                              "variant2": "neuweiler_rdb5",
                                              "variant3": "neuweiler",
                                              "variant4": "neuweiler"}

                    goal_prob_file_path, goal_prob_sem_file_path = get_lstm_results_path(
                        lstm_train_scenario[scenario_name],
                        input_type,
                        test_scenario_variants[
                            scenario_name] if args.lstm_train_scenario == "variants" else scenario_name,
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

                    temp_name = "neuweiler" if args.scenarios == "variant4" else scenario_name
                    true_goal_prob_sem = pd.read_csv(
                        results_dir + f'/{temp_name}_{model_name}_true_goal_prob_sem.csv')
                    true_goal_prob = pd.read_csv(results_dir + f'/{temp_name}_{model_name}_true_goal_prob.csv')
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
                label = label_map[
                    f"lstm_{input_type}{f'_{features_used_names}' if features_used_names != '' else ''}"]
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
