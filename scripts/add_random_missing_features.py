import pandas as pd
import numpy as np
from igp2.data import ScenarioConfig

from ogrit.core.base import get_data_dir, get_base_dir


scenarios = ['heckstrasse', 'bendplatz', 'frankenberg', 'round']

for scenario_name in scenarios:
    scenario_config = ScenarioConfig.load(get_base_dir() + f"/scenarios/configs/{scenario_name}.json")
    for episode_idx in range(len(scenario_config.episodes)):
        samples = pd.read_csv(get_data_dir() + f'/{scenario_name}_e{episode_idx}.csv')
        if scenario_name == 'round':
            samples['exit_number_missing'] = np.random.choice([True, False], size=samples.shape[0])
        else:
            samples['exit_number_missing'] = True
            samples['exit_number'] = 0
        samples['oncoming_vehicle_missing'] = np.random.choice([True, False], size=samples.shape[0])
        samples['vehicle_in_front_missing'] = np.random.choice([True, False], size=samples.shape[0])
        samples.to_csv(get_data_dir() + '/{}_e{}.csv'.format(scenario_name, episode_idx), index=False)
