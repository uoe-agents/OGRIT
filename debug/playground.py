import numpy as np

from igp2.agents.agentstate import AgentState
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map

from core.goal_generator import GoalGenerator

import matplotlib.pyplot as plt

xodr = "../maps/heckstrasse.xodr"
scenario_map = Map.parse_from_opendrive(xodr)

heading = np.deg2rad(-45)
speed = 5
time = 0
#position = np.array((28.9, -21.9))
position = np.array((31.3, -18.8))
#position = np.array((69.3, -42.9))
velocity = speed * np.array((np.cos(heading), np.sin(heading)))
acceleration = np.array((0, 0))

state = AgentState(time, position, velocity, acceleration, heading)

goal_generator = GoalGenerator()
goals = goal_generator.generate(scenario_map, state)
print(goals)

lanes = scenario_map.lanes_at(position)
print(lanes)
lane = lanes[0]
traversable_neighbours = lane.traversable_neighbours()

plot_map(scenario_map)
plt.plot(*lane.midline.coords.xy)
for l in traversable_neighbours:
    plt.plot(*l.midline.coords.xy)
    print(l)
plt.show()
