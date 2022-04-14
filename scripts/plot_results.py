import pandas as pd
import matplotlib.pyplot as plt
from grit.core.base import get_base_dir
import itertools

plt.style.use('ggplot')

model_names = ['prior_baseline', 'grit', 'generalised_grit', 'grit_uniform_prior']
scenario_names = ['heckstrasse', 'bendplatz', 'frankenberg', 'round']


# plot accuracy
fig, axes = plt.subplots(2, 2)

for scenario_idx, scenario_name in enumerate(scenario_names):
    ax = axes[scenario_idx % 2, scenario_idx // 2]
    plt.sca(ax)
    if scenario_idx % 2 == 1:
        plt.xlabel('fraction of trajectory observed')
    plt.title('Accuracy ({})'.format(scenario_name))
    marker = itertools.cycle(('^', '+', 'x', 'o', '*'))

    for model_name in model_names:
        accuracy_sem = pd.read_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_acc_sem.csv')
        accuracy = pd.read_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_acc.csv')

        plt.plot(accuracy.fraction_observed, accuracy.model_correct, label=model_name, marker=next(marker))
        plt.fill_between(accuracy_sem.fraction_observed, (accuracy + accuracy_sem).model_correct.to_numpy(),
                         (accuracy - accuracy_sem).model_correct.to_numpy(), alpha=0.2)
    plt.ylim([0, 1])
    plt.legend()


# plot normalised entropy
fig, axes = plt.subplots(2, 2)

for scenario_idx, scenario_name in enumerate(scenario_names):
    ax = axes[scenario_idx % 2, scenario_idx // 2]
    plt.sca(ax)
    if scenario_idx % 2 == 1:
        plt.xlabel('fraction of trajectory observed')
    plt.title('Normalised Entropy ({})'.format(scenario_name))
    marker = itertools.cycle(('^', '+', 'x', 'o', '*'))

    for model_name in model_names:
        entropy_norm_sem = pd.read_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_entropy_norm_sem.csv')
        entropy_norm = pd.read_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_entropy_norm.csv')

        plt.plot(entropy_norm.fraction_observed, entropy_norm.model_entropy_norm, label=model_name, marker=next(marker))
        plt.fill_between(entropy_norm_sem.fraction_observed, (entropy_norm + entropy_norm_sem).model_entropy_norm.to_numpy(),
                         (entropy_norm - entropy_norm_sem).model_entropy_norm.to_numpy(), alpha=0.2)
    plt.ylim([0, 1.1])
    plt.legend()

plt.show()
