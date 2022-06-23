import argparse

from grit.occlusion_detection.occlusion_detection_geometry import OcclusionDetector2D
from grit.core.base import create_folders


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default="bendplatz")
    parser.add_argument('--episode_idx', type=int, help='Name of scenario to process', default=0)
    parser.add_argument('--debug',
                        help="if set, we plot all the occlusions in a frame for each vehicle."
                             "If --debug_steps is also True, this takes precedence and --debug_steps will be"
                             "deactivated.",
                        action='store_true')

    parser.add_argument('--debug_steps',
                        help="if set, we plot the occlusions created by each obstacle. "
                             "If --debug is set, --debug_steps will be disabled.",
                        action='store_true')

    args = parser.parse_args()

    create_folders()

    print('scenario {} episode {}'.format(args.scenario, args.episode_idx))

    occlusion_detector = OcclusionDetector2D(args.scenario, args.episode_idx, debug=args.debug,
                                             debug_steps=args.debug_steps)

    occlusion_detector.extract_occlusions()
    print('finished scenario {} episode {}'.format(args.scenario, args.episode_idx))


if __name__ == '__main__':
    main()
