import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import matplotlib
import torch

import baselines.lstm.test as eval_lstm
from ogrit.core.base import get_data_dir, get_all_scenarios

colors = list(sns.color_palette("tab10"))
markers = ["o", "x", "s", "P"] # todo

# from: https://github.com/cbrewitt/av-goal-recognition/blob/lstm/evaluation/compare_models.py -- todo not publicly available
def draw_line_with_sem(group, ax, key, value, i=0):
    accuracy = group.mean()
    accuracy.index = np.arange(accuracy.index.size)
    accuracy_sem = group.std() / np.sqrt(group.count())
    accuracy_sem.index = np.arange(accuracy_sem.index.size)
    accuracy.rename(columns={key: value}).plot(ax=ax)
    ax.fill_between(accuracy_sem.index, (accuracy + accuracy_sem)[key].to_numpy(),
                     (accuracy - accuracy_sem)[key].to_numpy(), alpha=0.2)
    xs = range(0, accuracy.shape[0])
    ys = accuracy.values
    ax.scatter(xs, ys, marker=markers[i], color=colors[i], s=80)
    return ax


def main():
    scenarios = get_all_scenarios()  # todo: use get_all_scenarios / argument
    lstm_dataset = "trajectory"

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.size"] = 14

    sns.set_style("darkgrid")

    models, predictions = pickle.load(open(get_data_dir() + "grit_eval_data.p", "rb"))

    # todo: delete the following
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    nr_rows = math.ceil(len(scenarios)/4)
    nr_cols = min(len(scenarios), 4)
    plot_width = 5 * nr_cols
    plot_height = 5 * nr_rows

    # todo: Create a plot with 4 scenarios per row
    fig, ax = plt.subplots(nrows=nr_rows, ncols=len(scenarios), sharex=True, sharey=True,
                           figsize=(plot_width, plot_height))

    for scenario_idx, scenario_name in enumerate(scenarios):
        i = 0

        ax[scenario_idx].set_title(f"{scenario_name}".capitalize())

        # Plot OGRIT and prior
        for model_name, model in models.items():

            unique_samples = predictions[scenario_name][model_name]
            unique_samples['fraction_observed'] = round(unique_samples['fraction_observed'], 1) # todo: created bins every decimal digit
            fraction_observed_grouped = unique_samples[['model_correct', 'fraction_observed']].groupby(
                'fraction_observed')
            draw_line_with_sem(fraction_observed_grouped, ax[scenario_idx], i=i, key="model_correct",
                               value={"specialized_ogrit": "S-OGRIT", "occlusion_grit": "OGRIT", "grit": "GRIT"}[model_name])  # todo:
            i += 1

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
        ax[scenario_idx].plot(xs, true_goal_prob.true_goal_prob, label="LSTM",
                              marker=markers[i], color=colors[i])
        ax[scenario_idx].fill_between(xs,
                                      (true_goal_prob + true_goal_prob_sem).true_goal_prob.to_numpy(),
                                      (true_goal_prob - true_goal_prob_sem).true_goal_prob.to_numpy(), alpha=0.2)

        if scenario_idx % 4 == 0:
            # For each row, set the y label
            ax[scenario_idx].set_ylabel("Probability assigned to true goal")

        ax[scenario_idx].set_xlabel("Fraction of trajectory observed")
        ax[scenario_idx].get_legend().remove()

    ax[0].set_yticks(np.linspace(0.0, 1.0, fraction_observed_grouped.ngroups),
                     labels=[f"{v:.1f}" for v in np.linspace(0, 1, 11)])
    ax[0].set_ylim([-0.1, 1.1])
    ax[0].legend(loc="lower right")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)])
    fig.suptitle("Probability assigned to true goal vs fraction of trajectory observed")
    fig.tight_layout()

    # fig.savefig(get_base_dir() + f"/images/{scenario_name}_accuracy.pdf", bbox_inches='tight', pad_inches=0)
    plt.show() # todo

    fig, ax = plt.subplots(nrows=round(len(scenarios) / 4), ncols=min(len(scenarios), 4), figsize=(20, 5)) # todo: share many similarities with above

    for scenario_idx, scenario_name in enumerate(scenarios):
        i = 0

        ax[scenario_idx].set_title(f"{scenario_name}")

        for model_name, model in models.items():
            unique_samples = predictions[scenario_name][model_name]
            fraction_observed_grouped = unique_samples[['model_entropy_norm', 'fraction_observed']].groupby(
                'fraction_observed')
            draw_line_with_sem(fraction_observed_grouped, ax[scenario_idx], "model_entropy_norm",
                               {"specialized_ogrit": "S-OGRIT", "occlusion_grit": "OGRIT", "grit": "GRIT"}[model_name], i=i)
            i += 1

        for dataset in lstm_datasets:
            test_config = argparse.Namespace(**{
                "dataset": dataset,
                "shuffle": True,
                "scenario": scenario_name,
                "model_path": f"checkpoint/{scenario_name}_{dataset}_best.pt",
                "lstm_hidden_dim": 64, # todo: change as above
                "fc_hidden_dim": 725,
                "lstm_layers": 1,
                "step": 0.1
            })
            lstm_corrects, lstm_probs, _ = eval_lstm.main(test_config)
            xs = np.arange(lstm_corrects.shape[1])
            h_goals = lstm_probs * np.log2(lstm_probs)
            h_goals = -h_goals.sum(axis=-1)
            h_uniform = np.log2(lstm_probs.shape[-1])
            h = h_goals / h_uniform
            entropy = h.mean(0)
            sem = h.std(0) / np.sqrt(h.shape[0])
            ax[scenario_idx].plot(xs, entropy, label=f"LSTM")
            ax[scenario_idx].fill_between(xs, entropy + sem, entropy - sem, alpha=0.2)
            xs = range(0, accuracy.size)
            ys = entropy
        ax[scenario_idx].get_legend().remove()
        ax[scenario_idx].scatter(xs, ys, marker=markers[i], color=colors[i], s=80)
        ax[scenario_idx].set_yticks(np.linspace(-0.1, 1.1, 11))
        ax[scenario_idx].set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)])
        ax[scenario_idx].set_xticks(np.arange(fraction_observed_grouped.ngroups))
        ax[scenario_idx].set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 1, fraction_observed_grouped.ngroups)])
        ax[scenario_idx].set_xlabel('Fraction of trajectory observed')

    ax[0].set_ylabel('Normalised Entropy')
    fig.suptitle("Normalised entropy vs fraction of trajectory observed")
    plt.ylim([-0.1, 1.1])
    ax[0].legend(loc="lower right")
    fig.tight_layout()
    # fig.savefig(get_base_dir() + f"/images/{scenario_name}_entropy.pdf", bbox_inches='tight', pad_inches=0) todo
    plt.show()


if __name__ == '__main__':
    main()