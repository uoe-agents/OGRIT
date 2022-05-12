import numpy as np

from grit.decisiontree.dt_goal_recogniser import OcclusionGrit, GeneralisedGrit

model = OcclusionGrit.load('heckstrasse')
print(model.decision_trees['straight-on'].decision.threshold)

model = GeneralisedGrit.load('heckstrasse')
print(model.decision_trees['straight-on'].decision.threshold)