import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from ogrit.core.base import get_base_dir, get_all_scenarios, get_data_dir
from ogrit.core.data_processing import get_dataset
from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit, Grit, GeneralisedGrit, UniformPriorGrit, \
    OcclusionBaseline, NoPossiblyMissingFeaturesGrit, NoPossiblyMissingFeaturesGGrit, SpecializedOgrit, TruncatedGrit
from ogrit.goalrecognition.goal_recognition import PriorBaseline, UniformPriorBaseline


def drop_low_sample_agents(dataset, min_samples=2):
    unique_agent_pairs = dataset[['episode', 'agent_id', 'ego_agent_id', 'true_goal']].drop_duplicates()
    unique_agent_pairs['frame_count'] = 0
    unique_samples = dataset[['episode', 'agent_id', 'ego_agent_id', 'true_goal', 'frame_id']].drop_duplicates()
    vc = unique_samples.value_counts(['episode', 'agent_id', 'ego_agent_id'])
    vc = vc.to_frame().reset_index().rename(columns={0: 'sample_count'})
    new_dataset = dataset.merge(vc, on=['episode', 'agent_id', 'ego_agent_id'])
    new_dataset = new_dataset.loc[new_dataset.sample_count >= min_samples]
    return new_dataset


def evaluate_models(scenario_names=None, model_names=None, dataset_name='test', results_dir=None, data_dir=None,
                    predictions_dir=None, models_dir=None):

    if results_dir is None:
        results_dir = get_base_dir() + '/results/'
    if data_dir is None:
        data_dir = get_data_dir()
    if predictions_dir is None:
        predictions_dir = get_base_dir() + '/predictions/'
    if models_dir is None:
        models_dir = data_dir

    plt.style.use('ggplot')

    if model_names is None:
        model_names = ['generalised_grit', 'occlusion_grit']

    if scenario_names is None:
        scenario_names = get_all_scenarios()

    model_classes = {'prior_baseline': PriorBaseline,
                     'uniform_prior_baseline': UniformPriorBaseline,
                     'occlusion_grit': OcclusionGrit,
                     'grit': Grit,
                     'generalised_grit': GeneralisedGrit,
                     'grit_uniform_prior': UniformPriorGrit,
                     'occlusion_baseline': OcclusionBaseline,
                     'sogrit': SpecializedOgrit,
                     'truncated_grit': TruncatedGrit,
                     'no_possibly_missing_features_grit': NoPossiblyMissingFeaturesGrit,
                     'no_possibly_missing_features_ggrit': NoPossiblyMissingFeaturesGGrit}

    accuracies = pd.DataFrame(index=model_names, columns=scenario_names)
    accuracies_sem = pd.DataFrame(index=model_names, columns=scenario_names)
    cross_entropies = pd.DataFrame(index=model_names, columns=scenario_names)
    entropies = pd.DataFrame(index=model_names, columns=scenario_names)
    norm_entropies = pd.DataFrame(index=model_names, columns=scenario_names)
    avg_max_prob = pd.DataFrame(index=model_names, columns=scenario_names)
    avg_min_prob = pd.DataFrame(index=model_names, columns=scenario_names)
    true_goal_prob = pd.DataFrame(index=model_names, columns=scenario_names)

    predictions = {}

    for scenario_name in scenario_names:
        dataset = get_dataset(scenario_name, dataset_name, data_dir=data_dir)

        dataset = drop_low_sample_agents(dataset, 2)
        dataset_predictions = {}

        for model_name in model_names:
            print(model_name)
            model_class = model_classes[model_name]
            model = model_class.load(scenario_name, models_dir)
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

            unique_samples.to_csv(predictions_dir + '/{}_{}_{}.csv'.format(
                scenario_name, model_name, dataset_name), index=False)

        predictions[scenario_name] = dataset_predictions

    pickle.dump(({k: model_classes[k] for k in model_names}, predictions),
                open(data_dir + "/grit_eval_data.p", "wb")) # todo

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
            accuracy_sem.to_csv(results_dir + f'/{scenario_name}_{model_name}_acc_sem.csv')
            accuracy.to_csv(results_dir + f'/{scenario_name}_{model_name}_acc.csv')

            # save entropy norm
            fraction_observed_grouped = unique_samples[['model_entropy_norm', 'fraction_observed']].groupby(
                'fraction_observed')
            entropy_norm = fraction_observed_grouped.mean()
            entropy_norm_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
            entropy_norm_sem.to_csv(results_dir + f'/{scenario_name}_{model_name}_entropy_norm_sem.csv')
            entropy_norm.to_csv(results_dir + f'/{scenario_name}_{model_name}_entropy_norm.csv')

            # save true goal probability
            fraction_observed_grouped = unique_samples[['true_goal_prob', 'fraction_observed']].groupby('fraction_observed')
            true_goal_prob = fraction_observed_grouped.mean()
            true_goal_prob_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
            true_goal_prob_sem.to_csv(results_dir + f'/{scenario_name}_{model_name}_true_goal_prob_sem.csv')
            true_goal_prob.to_csv(results_dir + f'/{scenario_name}_{model_name}_true_goal_prob.csv')

            # save cross entropy
            fraction_observed_grouped = unique_samples[['cross_entropy', 'fraction_observed']].groupby('fraction_observed')
            cross_entropy = fraction_observed_grouped.mean()
            cross_entropy_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())
            cross_entropy_sem.to_csv(results_dir + f'/{scenario_name}_{model_name}_cross_entropy_sem.csv')
            cross_entropy.to_csv(results_dir + f'/{scenario_name}_{model_name}_cross_entropy.csv')
