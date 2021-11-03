import numpy as np
from igp2.agents.agentstate import AgentState
from igp2.opendrive.map import Map

from core.goal_generator import GoalGenerator, TypedGoal


def goals_close(g1: TypedGoal, g2: TypedGoal):
    return np.allclose(g1.goal.center, g2.goal.center) and g1.goal_type == g2.goal_type


def test_heckstrasse_north_west():
    xodr = "../maps/heckstrasse.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(-45)
    speed = 5
    time = 0
    position = np.array((28.9, -21.9))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, state)

    assert len(goals) == 2
    assert sum([g.goal_type == 'straight-on' and np.allclose(g.goal.center, (61.9, -47.3), atol=1) for g in goals]) == 1
    assert sum([g.goal_type == 'turn-left' and np.allclose(g.goal.center, (60.5, -18.7), atol=1) for g in goals]) == 1


def test_heckstrasse_south_east():
    xodr = "../maps/heckstrasse.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(135)
    speed = 5
    time = 0
    position = np.array((68.7, -42.9))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, state)

    assert len(goals) == 2
    assert sum([g.goal_type == 'straight-on' and np.allclose(g.goal.center, (35.5, -17.5), atol=1) for g in goals]) == 1
    assert sum([g.goal_type == 'turn-right' and np.allclose(g.goal.center, (60.5, -18.7), atol=1) for g in goals]) == 1


def test_heckstrasse_north_east():
    xodr = "../maps/heckstrasse.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(-135)
    speed = 5
    time = 0
    position = np.array((60.4, -15.1))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, state)

    assert len(goals) == 2
    assert sum([g.goal_type == 'turn-left' and np.allclose(g.goal.center, (62.1, -47.3), atol=1) for g in goals]) == 1
    assert sum([g.goal_type == 'turn-right' and np.allclose(g.goal.center, (35.1, -17.4), atol=1) for g in goals]) == 1
