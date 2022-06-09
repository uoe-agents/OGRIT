import numpy as np

from grit.decisiontree.dt_goal_recogniser import OcclusionGrit, GeneralisedGrit
import matplotlib.pyplot as plt
from igp2 import Map, plot_map
from igp2.data.scenario import ScenarioConfig

config = ScenarioConfig.load("scenarios/configs/frankenberg.json")
scenario = Map.parse_from_opendrive(f"scenarios/maps/frankenberg.xodr")
plot_map(scenario, scenario_config=config, plot_background=True)
plt.show()

# model = OcclusionGrit.load('heckstrasse')
# print(model.decision_trees['straight-on'].decision.threshold)
#
# model = GeneralisedGrit.load('heckstrasse')
# print(model.decision_trees['straight-on'].decision.threshold)