import pandas as pd
import argparse

from grit.core.base import get_data_dir, get_predictions_dir

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
base_model_name = "grit"


COLUMNS_TO_TAKE = ["episode", "agent_id", "ego_agent_id", "frame_id"]


for scenario_name in scenarios:

    samples_test_model = pd.read_csv(get_predictions_dir() + f'/original/{scenario_name}_{test_model_name}_all.csv')
    samples_base_model = pd.read_csv(get_predictions_dir() + f'/original/{scenario_name}_{base_model_name}_all.csv')

    samples_test_model["significant_sample"] = (samples_base_model["true_goal_prob"] -
                                                samples_test_model["true_goal_prob"]) > args.cut_off

    significant_samples = samples_test_model[samples_test_model["significant_sample"]][COLUMNS_TO_TAKE]

    significant_samples.to_csv(get_data_dir() + '/{}_{}_vs_{}.csv'.format(scenario_name, test_model_name, base_model_name), index=False)
