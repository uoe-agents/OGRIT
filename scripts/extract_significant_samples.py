import pandas as pd
import argparse

from ogrit.core.base import get_data_dir, get_predictions_dir
from scripts.train_generalised_decision_trees import main as train_generalised_tree
from scripts.train_occlusion_grit import main as train_ogrit
from scripts.evaluate_models_from_features import main as evaluate_models

parser = argparse.ArgumentParser(description='Extract the samples in which there is significant difference in the '
                                             'accuracy of GRIT with and without features that could be possibly'
                                             'missing')
parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
parser.add_argument('--cut_off', type=float, help='# How much more does the true goal probability have to be in the '
                                                   'predictions of the test model than the base one. Between 0 and 1',
                    default=0.01)

args = parser.parse_args()

scenarios = ['heckstrasse', 'bendplatz', 'frankenberg', 'round']

# Name of the models which we want to compare to the base model.
test_model_name = "no_possibly_missing_features_grit"
base_model_name = "ogrit"


COLUMNS_TO_TAKE = ["episode", "agent_id", "ego_agent_id", "frame_id"]


for scenario_name in scenarios:

    all_samples = pd.read_csv(get_data_dir() + f'/original/{scenario_name}_{test_model_name}.csv')
    samples_test_model = pd.read_csv(get_predictions_dir() + f'/original/{scenario_name}_{test_model_name}_all.csv')
    samples_base_model = pd.read_csv(get_predictions_dir() + f'/original/{scenario_name}_{base_model_name}_all.csv')

    samples_test_model["significant_sample"] = (samples_base_model["true_goal_prob"] -
                                                samples_test_model["true_goal_prob"]) > args.cut_off

    significant_samples = samples_test_model[samples_test_model["significant_sample"]]

    merged_samples = all_samples.merge(significant_samples, on=COLUMNS_TO_TAKE)

    for episode_idx in merged_samples["episode"].unique():
        scenario_samples = merged_samples[merged_samples["episode"] == episode_idx]
        scenario_samples.to_csv(get_data_dir() + f'/{scenario_name}_e{episode_idx}.csv', index=False)

    # Train the models
    train_generalised_tree()
    train_ogrit()
    evaluate_models()



