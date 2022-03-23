import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from core.base import get_base_dir
from core.data_processing import get_dataset
from decisiontree.dt_goal_recogniser import Grit, GeneralisedGrit, UniformPriorGrit, HandcraftedGoalTrees
from goalrecognition.goal_recognition import PriorBaseline


def main():
    plt.style.use('ggplot')

    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenario', type=str, help='Name of scenario to validate', default=None)
    parser.add_argument('--models', type=str, help='List of models, comma separated', default=None)
    args = parser.parse_args()

    if args.scenario is None:
        scenario_names = ['heckstrasse', 'bendplatz', 'frankenberg', 'round']
    else:
        scenario_names = [args.scenario]

    model_classes = {'prior_baseline': PriorBaseline,
                     #'handcrafted_trees': HandcraftedGoalTrees,
                     'grit': Grit,
                     'generalised_grit': GeneralisedGrit,
                     'grit_uniform_prior': UniformPriorGrit}

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
            accuracy = unique_samples.model_correct.mean()
            accuracies_sem.loc[model_name, scenario_name] = unique_samples.model_correct.sem()
            accuracies.loc[model_name, scenario_name] = accuracy
            cross_entropies.loc[model_name, scenario_name] = cross_entropy
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

    for scenario_name in scenario_names:
        for idx, model_name in enumerate(model_names):
            unique_samples = predictions[scenario_name][model_name]
            fraction_observed_grouped = unique_samples[['model_correct', 'fraction_observed']].groupby('fraction_observed')
            accuracy = fraction_observed_grouped.mean()
            accuracy_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())

            # save results
            accuracy_sem.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_acc_sem.csv')
            accuracy.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_acc.csv')

        for model_name in model_names:
            unique_samples = predictions[scenario_name][model_name]
            fraction_observed_grouped = unique_samples[['model_entropy_norm', 'fraction_observed']].groupby('fraction_observed')
            entropy_norm = fraction_observed_grouped.mean()
            entropy_norm_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
            # save results
            entropy_norm_sem.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_entropy_norm_sem.csv')
            entropy_norm.to_csv(get_base_dir() + f'/results/{scenario_name}_{model_name}_entropy_norm.csv')


if __name__ == '__main__':
    main()
