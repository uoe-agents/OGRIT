import numpy as np

from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit, GeneralisedGrit
import matplotlib.pyplot as plt
from igp2 import Map, plot_map
from igp2.data.scenario import ScenarioConfig

config = ScenarioConfig.load("scenarios/configs/heckstrasse.json")
scenario = Map.parse_from_opendrive(f"scenarios/maps/heckstrasse.xodr")
plot_map(scenario, scenario_config=config, plot_background=True)
plt.show()

for priority in scenario.junctions[0].priorities:
    print(priority)
    plot_map(scenario, scenario_config=config, plot_background=True)
    for l in priority.high.lanes.lane_sections[0].all_lanes:
        plt.plot(*l.midline.coords.xy, color='red')
    for l in priority.low.lanes.lane_sections[0].all_lanes:
        plt.plot(*l.midline.coords.xy, color='blue')
    plt.show()

# model = OcclusionGrit.load('heckstrasse')
# print(model.decision_trees['straight-on'].decision.threshold)
#
# model = GeneralisedGrit.load('heckstrasse')
# print(model.decision_trees['straight-on'].decision.threshold)