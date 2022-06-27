import pandas as pd
import numpy as np
from igp2.data import ScenarioConfig

from ogrit.core.base import get_data_dir, get_base_dir


scenarios = ['frankenberg', 'heckstrasse', 'bendplatz', 'round']

for scenario_name in scenarios:
    scenario_config = ScenarioConfig.load(get_base_dir() + f"/scenarios/configs/{scenario_name}.json")
    for episode_idx in range(len(scenario_config.episodes)):
        samples = pd.read_csv(get_data_dir() + f'/{scenario_name}_e{episode_idx}.csv')

        for target_vehicle_id in samples["agent_id"].unique():
            target_samples = samples[samples["agent_id"] == target_vehicle_id]

            for ego_agent_id in target_samples["ego_agent_id"].unique():

                # Store the id of the last frame in which both vehicles are alive at the same time.
                ego_target_pair_data = target_samples[target_samples["ego_agent_id"] == ego_agent_id]
                final_frame_id = max(ego_target_pair_data["frame_id"])
                samples.loc[samples.index[ego_target_pair_data.index], 'final_frame_id'] = final_frame_id

        samples["fraction_observed"] = (samples["frame_id"] - samples["initial_frame_id"]) / \
                                       (samples["final_frame_id"] - samples["initial_frame_id"])

        assert max(samples["fraction_observed"]) <= 1.00

        samples.to_csv(get_data_dir() + '/{}_e{}.csv'.format(scenario_name, episode_idx), index=False)
