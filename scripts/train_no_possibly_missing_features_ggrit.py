import argparse

from ogrit.core.data_processing import get_multi_scenario_dataset
from ogrit.decisiontree.dt_goal_recogniser import GeneralisedGrit, NoPossiblyMissingFeaturesGGrit
from ogrit.core.base import get_all_scenarios


def main():
    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenario', type=str, help='Name of scenario to validate', default=None)
    parser.add_argument('--dataset', type=str, help='Subset of data to train on', default='train')
    args = parser.parse_args()

    if args.scenario is None:
        scenario_names = get_all_scenarios()
    else:
        scenario_names = [args.scenario]

    dataset = get_multi_scenario_dataset(scenario_names, subset=args.dataset)

    ggrit = NoPossiblyMissingFeaturesGGrit.train(scenario_names,
                                 criterion='entropy',
                                 min_samples_leaf=10,
                                 max_depth=7,
                                 alpha=1, ccp_alpha=0.0001,
                                 dataset=dataset)
    ggrit.save()


if __name__ == '__main__':
    main()
