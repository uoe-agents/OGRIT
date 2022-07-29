import argparse
from datetime import datetime
from ogrit.core.data_processing import prepare_episode_dataset


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default="bendplatz")
    parser.add_argument('--episode_idx', type=int, help='Name of scenario to process', default=0)
    parser.add_argument('--no_indicator_features', help='If you don`t want to extract the indicator features',
                        action='store_false')

    args = parser.parse_args()

    start = datetime.now()
    prepare_episode_dataset((args.scenario, args.episode_idx, args.no_indicator_features))
    print(datetime.now() - start)


if __name__ == '__main__':
    main()
