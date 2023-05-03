import numpy as np
import pandas as pd
from igp2 import Map, PointGoal
from igp2.data import ScenarioConfig, InDScenario
import sys
from ogrit.core.base import get_scenarios_dir, set_working_dir
from ogrit.core.feature_extraction import FeatureExtractor
from ogrit.core.goal_generator import TypedGoal

episode_idx = sys.argv[1]

scenario_name = 'neuweiler'
scenario_map = Map.parse_from_opendrive(get_scenarios_dir() + f"maps/{scenario_name}.xodr")
scenario_config = ScenarioConfig.load(get_scenarios_dir() + f"configs/{scenario_name}.json")

set_working_dir()
scenario = InDScenario(scenario_config)

print(f'episode {episode_idx}')
feature_extractor = FeatureExtractor(scenario_map, scenario_name, episode_idx)

file_path = f'data/{scenario_name}_e{episode_idx}.csv'
samples = pd.read_csv(file_path)

roundabout_slip_road = []
roundabout_uturn = []
for index, row in samples.iterrows():
    exit_number = row['exit_number']
    goal_loc = scenario_config.goals[row['possible_goal']]
    goal = TypedGoal('roundabout-exit', PointGoal(np.array(goal_loc), 1.5), [])
    roundabout_uturn.append(feature_extractor.is_roundabout_uturn(exit_number))
    roundabout_slip_road.append(feature_extractor.slip_road(exit_number, goal))

samples['roundabout_slip_road'] = roundabout_slip_road
samples['roundabout_uturn'] = roundabout_uturn
samples.to_csv(file_path, index=False)
