import numpy as np

from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit, GeneralisedGrit
import matplotlib.pyplot as plt
from igp2 import Map, plot_map
from igp2.data.scenario import ScenarioConfig, InDScenario

scenario_name = 'neukoellnerstrasse'
#scenario_name = 'heckstrasse'
episode_idx = 5
config = ScenarioConfig.load(f"scenarios/configs/{scenario_name}.json")
scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{scenario_name}.xodr")
scenario = InDScenario(config)
episode = scenario.load_episode(episode_idx)

plot_map(scenario_map, scenario_config=config, plot_background=True)

for agent in episode.agents.values():
    if agent.metadata.agent_type == 'car':
        path = agent.trajectory.path
        plt.plot(path[:, 0], path[:, 1], linewidth=0.5)

plt.show()



# for priority in scenario.junctions[0].priorities:
#     print(priority)
#     plot_map(scenario, scenario_config=config, plot_background=True)
#     for l in priority.high.lanes.lane_sections[0].all_lanes:
#         plt.plot(*l.midline.coords.xy, color='red')
#     for l in priority.low.lanes.lane_sections[0].all_lanes:
#         plt.plot(*l.midline.coords.xy, color='blue')
#     plt.show()

# model = OcclusionGrit.load('heckstrasse')
# print(model.decision_trees['straight-on'].decision.threshold)
#
# model = GeneralisedGrit.load('heckstrasse')
# print(model.decision_trees['straight-on'].decision.threshold)