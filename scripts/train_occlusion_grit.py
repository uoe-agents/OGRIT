import argparse

from grit.decisiontree.dt_goal_recogniser import OcclusionGrit


def main():
    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
    parser.add_argument('--scenario', type=str, help='Name of scenario to validate', default=None)
    args = parser.parse_args()

    if args.scenario is None:
        scenario_names = ['heckstrasse', 'bendplatz', 'frankenberg', 'round']
    else:
        scenario_names = [args.scenario]

    grit = OcclusionGrit.train(scenario_names,
                                 criterion='entropy',
                                 min_samples_leaf=10,
                                 max_depth=7,
                                 alpha=1, ccp_alpha=0.0001)
    grit.save()


if __name__ == '__main__':
    main()
