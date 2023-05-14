import argparse
from datetime import datetime

from ogrit.core.base import set_working_dir
from ogrit.core.data_processing import prepare_episode_dataset
from ogrit.core.logger import logger


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default="heckstrasse")
    parser.add_argument('--episode_idx', type=int, help='Name of scenario to process', default=0)
    parser.add_argument('--no_indicator_features', help='If you don`t want to extract the indicator features',
                        action='store_false')

    args = parser.parse_args()

    logger.info(f'Start preprocessing for scenario {args.scenario} episode {args.episode_idx}')
    start = datetime.now()
    prepare_episode_dataset((args.scenario, args.episode_idx, args.no_indicator_features))
    logger.info(f"Preprocessing for scenario {args.scenario} episode {args.episode_idx} took {datetime.now() - start}")


if __name__ == '__main__':
    set_working_dir()
    main()
