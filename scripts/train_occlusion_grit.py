import argparse

from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit
from ogrit.core.base import get_all_scenarios


def main():
    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenarios', type=str, help='Name of scenarios to validate, comma separated', default=None)
    parser.add_argument('--subset', type=str, help='Subset of data to train on', default='train')
    parser.add_argument('--suffix', type=str, help='Suffix to add to model name', default='')
    args = parser.parse_args()

    if args.scenarios is None:
        scenario_names = get_all_scenarios()
    else:
        scenario_names = args.scenarios.split(',')

    model = OcclusionGrit.train(scenario_names,
                                criterion='entropy', min_samples_leaf=10, max_depth=7,
                                alpha=1, ccp_alpha=0.0001, balance_scenarios=True)
    model.save(name_suffix=args.suffix)


if __name__ == '__main__':
    main()
