import argparse
from multiprocessing import Pool

from ogrit.occlusion_detection.occlusion_detection_geometry import OcclusionDetector2D
from ogrit.core.base import create_folders, set_working_dir, get_all_scenarios
from igp2.data import ScenarioConfig


def prepare_episode_occlusion_dataset(params):
    scenario_name, episode_idx, debug, save_format, road_occlusions = params

    print('scenario {} episode {}'.format(scenario_name, episode_idx))

    occlusion_detector = OcclusionDetector2D(scenario_name, episode_idx, debug=debug, road_occlusions=road_occlusions)

    occlusion_detector.extract_occlusions(save_format)
    print('finished scenario {} episode {}'.format(scenario_name, episode_idx))


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
    parser.add_argument('--workers', type=int, help='Number of multiprocessing workers', default=8)
    parser.add_argument('--debug',
                        help="if set, we plot all the occlusions in a frame for each vehicle."
                             "If --debug_steps is also True, this takes precedence and --debug_steps will be"
                             "deactivated.",
                        action='store_true')
    parser.add_argument('--road_occlusions',
                        help="if set, we also store the occlusions on each road.", action='store_true')
    parser.add_argument('--save_format', type=str, help='Format in which to save the occlusion data. Either "json" '
                                                        'or "p" for pickle', default="p")

    args = parser.parse_args()

    create_folders()

    if args.scenario is None:
        scenarios = get_all_scenarios()
    else:
        scenarios = [args.scenario]

    params_list = []
    for scenario_name in scenarios:
        scenario_config = ScenarioConfig.load(f"scenarios/configs/{scenario_name}.json")
        for episode_idx in range(len(scenario_config.episodes)):
            params_list.append((scenario_name, episode_idx, args.debug, args.save_format, args.road_occlusions))

    with Pool(args.workers) as p:
        p.map(prepare_episode_occlusion_dataset, params_list)


if __name__ == '__main__':
    set_working_dir()
    main()
