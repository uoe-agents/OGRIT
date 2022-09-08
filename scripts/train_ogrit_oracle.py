import argparse

from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit, OgritOracle
from ogrit.core.base import get_all_scenarios


def main():
    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenarios', type=str, help='Name of scenarios to validate, comma separated', default=None)
    args = parser.parse_args()

    if args.scenarios is None:
        scenario_names = get_all_scenarios()
    else:
        scenario_names = args.scenarios.split(',')

    model = OgritOracle.train(scenario_names,
                                criterion='entropy', min_samples_leaf=10, max_depth=7,
                                alpha=1, ccp_alpha=0.0001, balance_scenarios=True)
    model.save()


if __name__ == '__main__':
    main()
