import pandas as pd
import argparse
from pathlib import Path

from ogrit.core.base import get_data_dir, get_predictions_dir, get_base_dir
from ogrit.core.data_processing import get_dataset, get_multi_scenario_dataset
from ogrit.decisiontree.dt_goal_recogniser import GeneralisedGrit, OcclusionGrit, NoPossiblyMissingFeaturesGGrit
from ogrit.evaluation.model_evaluation import evaluate_models

parser = argparse.ArgumentParser(description='Extract the samples in which there is significant difference in the '
                                             'accuracy of GRIT with and without features that could be possibly'
                                             'missing')
parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
parser.add_argument('--cut_off', type=float, help='# How much more does the true goal probability have to be in the '
                                                   'predictions of the test model than the base one. Between 0 and 1',
                    default=0.01)

args = parser.parse_args()

scenarios = ['heckstrasse', 'bendplatz', 'frankenburg']

# Name of the models which we want to compare to the base model.
test_model_name = "no_possibly_missing_features_ggrit"
base_model_name = "generalised_grit"
predictions_dir_all = get_predictions_dir() + '/models_all_data/'
Path(predictions_dir_all).mkdir(parents=True, exist_ok=True)

training_set = get_multi_scenario_dataset(scenarios, 'all')
ggrit = GeneralisedGrit.train(scenario_names=scenarios,
                              dataset=training_set,
                              criterion='entropy',
                              min_samples_leaf=10,
                              max_depth=7,
                              alpha=1, ccp_alpha=0.0001)
ggrit.save()

ggrit_no_missing_features = NoPossiblyMissingFeaturesGGrit.train(
                                             scenario_names=scenarios,
                                             criterion='entropy',
                                             min_samples_leaf=10,
                                             max_depth=7,
                                             alpha=1, ccp_alpha=0.0001,
                                             dataset=training_set)
ggrit_no_missing_features.save()
evaluate_models(model_names=[test_model_name, base_model_name], dataset_name='all',
                predictions_dir=predictions_dir_all, scenario_names=scenarios)


COLUMNS_TO_TAKE = ["episode", "agent_id", "ego_agent_id", "frame_id"]


data_dir = get_data_dir() + "/occlusion_subset/"
Path(data_dir).mkdir(parents=True, exist_ok=True)
results_dir = get_base_dir() + "/predictions/occlusion_subset/"
Path(results_dir).mkdir(parents=True, exist_ok=True)


for scenario_name in scenarios:

    all_samples = get_dataset(scenario_name, 'all')
    samples_test_model = pd.read_csv(get_predictions_dir() + f'/models_all_data/{scenario_name}_{test_model_name}_all.csv')
    samples_base_model = pd.read_csv(get_predictions_dir() + f'/models_all_data/{scenario_name}_{base_model_name}_all.csv')

    samples_test_model["significant_sample"] = (samples_base_model["true_goal_prob"] -
                                                samples_test_model["true_goal_prob"]) > args.cut_off

    significant_samples = samples_test_model[samples_test_model["significant_sample"]]

    merged_samples = all_samples.merge(significant_samples[COLUMNS_TO_TAKE], on=COLUMNS_TO_TAKE)

    for episode_idx in merged_samples["episode"].unique():
        scenario_samples = merged_samples[merged_samples["episode"] == episode_idx]
        scenario_samples.to_csv(data_dir + f'/{scenario_name}_e{episode_idx}.csv', index=False)

# Train the models
training_set = get_multi_scenario_dataset(scenarios, 'train', data_dir)
ggrit = GeneralisedGrit.train(scenarios,
                              dataset=training_set,
                              criterion='entropy',
                              min_samples_leaf=10,
                              max_depth=7,
                              alpha=1, ccp_alpha=0.0001)
ggrit.save(data_dir)

ogrit = OcclusionGrit.train(scenarios,
                            dataset=training_set,
                            criterion='entropy',
                            min_samples_leaf=10,
                            max_depth=7,
                            alpha=1, ccp_alpha=0.0001)
ogrit.save(data_dir)
evaluate_models(model_names=['generalised_grit', 'occlusion_baseline', 'occlusion_grit'],
                data_dir=data_dir, results_dir=results_dir, scenario_names=scenarios)

evaluate_models(model_names=['generalised_grit', 'no_possibly_missing_features_ggrit'], dataset_name='all',
                data_dir=data_dir, results_dir=results_dir, scenario_names=scenarios)



