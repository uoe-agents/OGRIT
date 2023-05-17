import numpy as np

from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit, GeneralisedGrit
import matplotlib.pyplot as plt
from igp2 import Map, plot_map
from igp2.data.scenario import ScenarioConfig, InDScenario

scenario_name = 'rdb7'
episode_idx = 0
config = ScenarioConfig.load(f"scenarios/configs/{scenario_name}.json")
scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{scenario_name}.xodr")
scenario = InDScenario(config)

plot_map(scenario_map, scenario_config=config, plot_background=False)

# count = 0
# episode = scenario.load_episode(episode_idx)
# for agent in episode.agents.values():
#     if agent.metadata.agent_type == 'car':
#         path = agent.trajectory.path
#         plt.plot(path[:, 0], path[:, 1], linewidth=0.5)
#         print(agent.agent_id)
#         count += 1
#     if count >= 100:
#         break

# lanes = scenario_map.lanes_at((24.0, -46.9), max_distance=0.01)
# print(lanes)
# lane = lanes[0]
# lane = scenario_map.get_lane(10, -1, 0)
# x, y = lane.boundary.exterior.xy
# plt.plot(x, y)
#
for x, y in config.goals:
    plt.plot(x, y, 'ro')

plt.show()
