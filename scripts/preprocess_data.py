import argparse
from multiprocessing import Pool

from igp2.data import ScenarioConfig

from ogrit.core.base import get_all_scenarios, set_working_dir
from ogrit.core.data_processing import prepare_episode_dataset


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
    parser.add_argument('--workers', type=int, help='Number of multiprocessing workers', default=2)
    parser.add_argument('--update_hz', type=int, help='take a sample every --update_hz frames in the original episode '
                                                      'frames (e.g., if 25, then take one frame per second)',
                        default=25)
    parser.add_argument('--no_indicator_features', help='If you don`t want to extract the indicator features',
                        action='store_true')

    args = parser.parse_args()

    if args.scenario is None:
        scenarios = get_all_scenarios()
    else:
        scenarios = [args.scenario]

    params_list = []
    for scenario_name in scenarios:
        scenario_config = ScenarioConfig.load(f"scenarios/configs/{scenario_name}.json")
        for episode_idx in range(len(scenario_config.episodes)):
            if args.no_indicator_features:
                # We want to extract the indicator features on top of the base features.
                params_list.append((scenario_name, episode_idx, False, args.update_hz))
            else:
                params_list.append((scenario_name, episode_idx, True, args.update_hz))

    with Pool(args.workers) as p:
        p.map(prepare_episode_dataset, params_list)


if __name__ == '__main__':
    set_working_dir()
    main()
