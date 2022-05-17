import numpy as np

from grit.decisiontree.dt_goal_recogniser import OcclusionGrit, GeneralisedGrit
import matplotlib.pyplot as plt
from igp2 import Map, plot_map
scenario = Map.parse_from_opendrive(f"scenarios/maps/heckstrasse.xodr")
plot_map(scenario)
plt.show()

# model = OcclusionGrit.load('heckstrasse')
# print(model.decision_trees['straight-on'].decision.threshold)
#
# model = GeneralisedGrit.load('heckstrasse')
# print(model.decision_trees['straight-on'].decision.threshold)