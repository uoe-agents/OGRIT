import argparse
from multiprocessing import Pool

from igp2.data import ScenarioConfig

from grit.core.data_processing import prepare_episode_dataset


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
    parser.add_argument('--workers', type=int, help='Number of multiprocessing workers', default=4)
    args = parser.parse_args()

    if args.scenario is None:
        scenarios = ['heckstrasse', 'bendplatz', 'frankenberg', 'round']
    else:
        scenarios = [args.scenario]

    params_list = []
    for scenario_name in scenarios:
        scenario_config = ScenarioConfig.load(f"scenarios/configs/{scenario_name}.json")
        for episode_idx in range(len(scenario_config.episodes)):
            params_list.append((scenario_name, episode_idx))

    #prepare_episode_dataset(('frankenberg', 7))

    with Pool(args.workers) as p:
        p.map(prepare_episode_dataset, params_list)


if __name__ == '__main__':
    main()
