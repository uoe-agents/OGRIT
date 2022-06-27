import numpy as np

from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit, GeneralisedGrit
import matplotlib.pyplot as plt
from igp2 import Map, plot_map
from igp2.data.scenario import ScenarioConfig

config = ScenarioConfig.load("scenarios/configs/heckstrasse.json")
scenario = Map.parse_from_opendrive(f"scenarios/maps/heckstrasse.xodr")
plot_map(scenario, scenario_config=config, plot_background=True)
plt.show()

print(scenario.lanes_at((19.61, -13.96)))
print(scenario.lanes_at((12.25, -5.08)))
print(scenario.lanes_at((51.29, -22.03)))

# model = OcclusionGrit.load('heckstrasse')
# print(model.decision_trees['straight-on'].decision.threshold)
#
# model = GeneralisedGrit.load('heckstrasse')
# print(model.decision_trees['straight-on'].decision.threshold)