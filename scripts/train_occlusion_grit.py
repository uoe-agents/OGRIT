import argparse

from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit
from ogrit.core.base import get_all_scenarios


def main():
    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenario', type=str, help='Name of scenario to validate', default=None)
    args = parser.parse_args()

    if args.scenario is None:
        scenario_names = get_all_scenarios()
    else:
        scenario_names = [args.scenario]

    grit = OcclusionGrit.train(scenario_names,
                                 criterion='entropy', min_samples_leaf=10, max_depth=7,
                                 alpha=1, ccp_alpha=0.1, balance_scenarios=True)
    grit.save()


if __name__ == '__main__':
    main()
