import pandas as pd
import numpy as np
from igp2.data import ScenarioConfig

from grit.core.base import get_data_dir, get_base_dir

scenario_name = 'round'
scenario_config = ScenarioConfig.load(get_base_dir() + f"/scenarios/configs/{scenario_name}.json")
for episode_idx in range(len(scenario_config.episodes)):
    samples = pd.read_csv(get_data_dir() + f'/{scenario_name}_e{episode_idx}.csv')
    samples['exit_number_missing'] = np.random.choice([True, False], size=samples.shape[0])
    samples.to_csv(get_data_dir() + '/{}_e{}.csv'.format(scenario_name, episode_idx), index=False)
