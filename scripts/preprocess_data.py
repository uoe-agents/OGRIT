import argparse
from multiprocessing import Pool

from igp2.data import ScenarioConfig

from grit.core.data_processing import prepare_episode_dataset


def main():

    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
    parser.add_argument('--workers', type=int, help='Number of multiprocessing workers', default=8)
    parser.add_argument('--extract_indicator_features', help='If you want to extract the indicator features',
                        action='store_true')
    args = parser.parse_args()

    iterate_through_scenarios(prepare_episode_dataset, args.scenario, args.workers,
                              args.extract_indicator_features)


def iterate_through_scenarios(function, scenario, workers, *boolean_flag):
    """
    Args:
        function: the method to call on each of the episodes of the scenario.
                  The argument must be a tuple of the type `(scenario_name, episode_idx)` or
                  `(scenario_name, episode_idx, boolean_argument)`
        scenario: what scenario to perform the function on.
        workers:  number of parallel workers doing the task.
    """

    if scenario is None:
        scenarios = ['heckstrasse', 'bendplatz', 'frankenberg', 'round']
    else:
        scenarios = [scenario]

    params_list = []
    for scenario_name in scenarios:
        scenario_config = ScenarioConfig.load(f"scenarios/configs/{scenario_name}.json")
        for episode_idx in range(len(scenario_config.episodes)):
            if boolean_flag:
                params_list.append((scenario_name, episode_idx, boolean_flag))
            else:
                params_list.append((scenario_name, episode_idx))

    with Pool(workers) as p:
        p.map(function, params_list)


if __name__ == '__main__':
    prepare_episode_dataset(("frankenberg", 5, True))
