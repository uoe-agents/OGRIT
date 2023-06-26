# Generate a random dataset split for the training and testing scenarios
import json

import numpy as np
from igp2.data import ScenarioConfig

from ogrit.core.base import get_map_configs_path, get_dataset_split_path

for scenario_name in ["rdb1", "rdb2", "rdb3", "rdb4", "rdb5", "rdb6", "rdb7"]:
    # Load the config file and get nr of episodes
    scenario_config = ScenarioConfig.load(get_map_configs_path(scenario_name))
    nr_episodes = len(scenario_config.config_dict["episodes"])

    # Generate a random split 60% training, 20% validation, 20% testing
    train_split = 0.6
    val_split = 0.2
    test_split = 0.2

    # Get the number of episodes for each split
    nr_train_episodes = int(train_split * nr_episodes)
    nr_val_episodes = int(val_split * nr_episodes)
    nr_test_episodes = int(test_split * nr_episodes)

    # Generate the split
    train_episodes = np.random.choice(range(nr_episodes), nr_train_episodes, replace=False)
    val_episodes = np.random.choice([episode for episode in range(nr_episodes) if
                                     episode not in train_episodes], nr_val_episodes, replace=False)
    test_episodes = np.random.choice([episode for episode in range(nr_episodes) if
                                      episode not in train_episodes and episode not in val_episodes],
                                     nr_test_episodes, replace=False)

    # Sort the episodes
    train_episodes.sort()
    val_episodes.sort()
    test_episodes.sort()

    print(f"Scenario: {scenario_name}")
    print(f"Train episodes: {train_episodes}")
    print(f"Val episodes: {val_episodes}")
    print(f"Test episodes: {test_episodes}")
    print("--------------------")

    # Save the split to OGRIT/ogrit/core/dataset_split.json
    with open(get_dataset_split_path(), 'r+') as f:
        data = json.load(f)
        a = {"train": list(train_episodes), "valid": list(val_episodes), "test": list(test_episodes),
             "all": list(range(nr_episodes))}
        data[scenario_name] = a

        f.seek(0)  # reset file position to the beginning.
        json.dump(data, f, indent=2, default=int)
        f.truncate()  # remove remaining part
