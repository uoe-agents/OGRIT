import argparse

from ogrit.core.data_processing import get_multi_scenario_dataset
from ogrit.decisiontree.dt_goal_recogniser import GeneralisedGrit
from ogrit.core.base import get_all_scenarios


def main():
    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenarios', type=str, help='Name of scenario to validate', default=None)
    parser.add_argument('--dataset', type=str, help='Subset of data to train on', default='train')
    args = parser.parse_args()

    if args.scenarios is None:
        scenario_names = get_all_scenarios()
    else:
        scenario_names = args.scenarios.split(',')

    dataset = get_multi_scenario_dataset(scenario_names, subset=args.dataset)

    ggrit = GeneralisedGrit.train(scenario_names,
                                  criterion='entropy',
                                  min_samples_leaf=10,
                                  max_depth=7,
                                  alpha=1, ccp_alpha=0.0001,
                                  dataset=dataset,
                                  balance_scenarios=True)
    ggrit.save()


if __name__ == '__main__':
    main()
