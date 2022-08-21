import argparse
from datetime import datetime

from ogrit.occlusion_detection.occlusion_detection_geometry import OcclusionDetector2D
from ogrit.core.base import create_folders, set_working_dir


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default="bendplatz")
    parser.add_argument('--episode_idx', type=int, help='Name of scenario to process', default=0)
    parser.add_argument('--debug',
                        help="if set, we plot all the occlusions in a frame for each vehicle."
                             "If --debug_steps is also True, this takes precedence and --debug_steps will be"
                             "deactivated.",
                        action='store_true')

    parser.add_argument('--save_format', type=str, help='Format in which to save the occlusion data. Either "json" '
                                                        'or "p" for pickle', default="p")

    args = parser.parse_args()

    create_folders()

    print('scenario {} episode {}'.format(args.scenario, args.episode_idx))

    occlusion_detector = OcclusionDetector2D(args.scenario, args.episode_idx, debug=args.debug)

    start = datetime.now()
    occlusion_detector.extract_occlusions(save_format=args.save_format)
    print(datetime.now() - start)


if __name__ == '__main__':
    set_working_dir()
    main()
