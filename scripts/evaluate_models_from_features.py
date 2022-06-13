import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from grit.core.base import get_base_dir
from grit.core.data_processing import get_dataset
from grit.decisiontree.dt_goal_recogniser import Grit, GeneralisedGrit, UniformPriorGrit, OcclusionGrit, \
    OcclusionBaseline
from grit.goalrecognition.goal_recognition import PriorBaseline, UniformPriorBaseline


def main():
    plt.style.use('ggplot')

    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenario', type=str, help='Name of scenario to validate', default=None)
    parser.add_argument('--models', type=str, help='List of models, comma separated', default=None)
    args = parser.parse_args()

    if args.scenario is None:
        scenario_names = ['heckstrasse', 'bendplatz', 'frankenberg']#, 'round']
    else:
        scenario_names = [args.scenario]

    model_classes = {'prior_baseline': PriorBaseline,
                     'uniform_prior_baseline': UniformPriorBaseline,
                     'occlusion_grit': OcclusionGrit,
                     'grit': Grit,
                     'generalised_grit': GeneralisedGrit,
                     'grit_uniform_prior': UniformPriorGrit,
                     'occlusion_baseline': OcclusionBaseline}

    if args.models is None:
        model_names = list(model_classes.keys())
    else:
        model_names = args.models.split(',')

    accuracies = pd.DataFrame(index=model_names, columns=scenario_names)
    accuracies_sem = pd.DataFrame(index=model_names, columns=scenario_names)
    cross_entropies = pd.DataFrame(index=model_names, columns=scenario_names)
    entropies = pd.DataFrame(index=model_names, columns=scenario_names)
    norm_entropies = pd.DataFrame(index=model_names, columns=scenario_names)
    avg_max_prob = pd.DataFrame(index=model_names, columns=scenario_names)
    avg_min_prob = pd.DataFrame(index=model_names, columns=scenario_names)
    true_goal_prob = pd.DataFrame(index=model_names, columns=scenario_names)

    predictions = {}
    dataset_name = 'test'

    for scenario_name in scenario_names:
        dataset = get_dataset(scenario_name, dataset_name)
        dataset_predictions = {}

        for model_name in model_names:
            model_class = model_classes[model_name]
            model = model_class.load(scenario_name)
            unique_samples = model.batch_goal_probabilities(dataset)
            unique_samples['model_correct'] = (unique_samples['model_prediction']
                                               == unique_samples['true_goal'])
            cross_entropy = -np.mean(np.log(unique_samples.loc[
                                                        unique_samples.model_probs != 0, 'model_probs']))

            true_goal_prob.loc[model_name, scenario_name] = unique_samples.true_goal_prob.mean()
            accuracy = unique_samples.model_correct.mean()
            accuracies_sem.loc[model_name, scenario_name] = unique_samples.model_correct.sem()
            accuracies.loc[model_name, scenario_name] = accuracy
            cross_entropies.loc[model_name, scenario_name] = unique_samples.cross_entropy.mean()
            entropies.loc[model_name, scenario_name] = unique_samples.model_entropy.mean()
            norm_entropies.loc[model_name, scenario_name] = unique_samples.model_entropy_norm.mean()
            avg_max_prob.loc[model_name, scenario_name] = unique_samples.max_probs.mean()
            avg_min_prob.loc[model_name, scenario_name] = unique_samples.min_probs.mean()
            dataset_predictions[model_name] = unique_samples
            print('{} accuracy: {:.3f}'.format(model_name, accuracy))
            print('{} cross entropy: {:.3f}'.format(model_name, cross_entropy))

            unique_samples.to_csv(get_base_dir() + '/predictions/{}_{}_{}.csv'.format(
                scenario_name, model_name, dataset_name), index=False)

        predictions[scenario_name] = dataset_predictions

    print('accuracy:')
    print(accuracies)
    print('accuracy sem:')
    print(accuracies_sem)
    print('\ncross entropy:')
    print(cross_entropies)
    print('\nentropy:')
    print(entropies)
    print('\nnormalised entropy:')
    print(norm_entropies)
    print('\naverage max probability:')
    print(avg_max_prob)
    print('\naverage min probability:')
    print(avg_min_prob)
    print('\ntrue goal probability:')
    print(true_goal_prob)

    for scenario_name in scenario_names:
        for model_name in model_names:
            unique_samples = predictions[scenario_name][model_name]
            unique_samples['fraction_observed'] = unique_samples['fraction_observed'].round(1)
            # save accuracy
            fraction_observed_grouped = unique_samples[['model_correct', 'fraction_observed']].groupby('fraction_observed')
            accuracy = fraction_observed_grouped.mean()
            accuracy_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
            accuracy_sem.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_acc_sem.csv')
            accuracy.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_acc.csv')

            # save entropy norm
            fraction_observed_grouped = unique_samples[['model_entropy_norm', 'fraction_observed']].groupby('fraction_observed')
            entropy_norm = fraction_observed_grouped.mean()
            entropy_norm_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
            entropy_norm_sem.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_entropy_norm_sem.csv')
            entropy_norm.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_entropy_norm.csv')

            # save true goal probability
            fraction_observed_grouped = unique_samples[['true_goal_prob', 'fraction_observed']].groupby('fraction_observed')
            true_goal_prob = fraction_observed_grouped.mean()
            true_goal_prob_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
            true_goal_prob_sem.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_true_goal_prob_sem.csv')
            true_goal_prob.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_true_goal_prob.csv')

            # save cross entropy
            fraction_observed_grouped = unique_samples[['cross_entropy', 'fraction_observed']].groupby('fraction_observed')
            cross_entropy = fraction_observed_grouped.mean()
            cross_entropy_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
            cross_entropy_sem.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_cross_entropy_sem.csv')
            cross_entropy.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_cross_entropy.csv')


if __name__ == '__main__':
    main()
