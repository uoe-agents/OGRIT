import argparse
import json

from ogrit.core.base import get_dt_config_dir, get_all_scenarios
from ogrit.core.data_processing import get_dataset
from ogrit.decisiontree.dt_goal_recogniser import NoPossiblyMissingFeaturesGrit


def main():
    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenario', type=str, help='Name of scenario to validate', default=None)
    parser.add_argument('--dataset', type=str, help='Subset of data to train on', default='train')
    args = parser.parse_args()

    if args.scenario is None:
        scenario_names = get_all_scenarios()
    else:
        scenario_names = [args.scenario]

    for scenario_name in scenario_names:
        with open(get_dt_config_dir() + scenario_name + '.json') as f:
            dt_params = json.load(f)
        training_set = get_dataset(scenario_name, args.dataset)
        model = NoPossiblyMissingFeaturesGrit.train(scenario_name, training_set=training_set, **dt_params)
        model.save(scenario_name)


if __name__ == '__main__':
    main()
