import numpy as np
from igp2 import AgentState
from igp2.opendrive.map import Map
from igp2.trajectory import VelocityTrajectory
from igp2.util import Circle

from ogrit.core.goal_generator import GoalGenerator


def goal_in_list(goals, goal_type, goal_center):
    return sum([g.goal_type == goal_type and np.allclose(g.goal.center, goal_center, atol=1) for g in goals]) == 1


def test_heckstrasse_north_west():
    xodr = "../scenarios/maps/heckstrasse.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(-45)
    speed = 5
    time = 0
    position = np.array((28.9, -21.9))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 2
    assert goal_in_list(goals, 'straight-on', (61.9, -47.3))
    assert goal_in_list(goals, 'turn-left', (60.5, -18.7))


def test_heckstrasse_south_east():
    xodr = "../scenarios/maps/heckstrasse.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(135)
    speed = 5
    time = 0
    position = np.array((68.7, -42.9))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 2
    assert sum([g.goal_type == 'straight-on' and np.allclose(g.goal.center, (35.5, -17.5), atol=1) for g in goals]) == 1
    assert sum([g.goal_type == 'turn-right' and np.allclose(g.goal.center, (60.5, -18.7), atol=1) for g in goals]) == 1


def test_heckstrasse_north_east():
    xodr = "../scenarios/maps/heckstrasse.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(-135)
    speed = 5
    time = 0
    position = np.array((60.4, -15.1))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 2
    assert goal_in_list(goals, 'turn-left', (61.9, -47.3))
    assert goal_in_list(goals, 'turn-right', (35.1, -17.4))


def test_bendplatz_south_west():
    xodr = "../scenarios/maps/bendplatz.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(45)
    speed = 5
    time = 0
    position = np.array((48.5, -43.8))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 3
    assert goal_in_list(goals, 'turn-left', (49.3, -21.0))
    assert goal_in_list(goals, 'turn-right', (63.0, -45.4))
    assert goal_in_list(goals, 'straight-on', (66.8, -22.2))


def test_bendplatz_north_east():
    xodr = "../scenarios/maps/bendplatz.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(-135)
    speed = 5
    time = 0
    position = np.array((65.4, -18.1))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 3
    assert goal_in_list(goals, 'turn-left', (62.3, -45.6))
    assert goal_in_list(goals, 'turn-right', (49.3, -21.))
    assert goal_in_list(goals, 'straight-on', (46.7, -40.8))


def test_bendplatz_northwest():
    xodr = "../scenarios/maps/bendplatz.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(-45)
    speed = 5
    time = 0
    position = np.array((25.5, -3.5))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 3
    assert goal_in_list(goals, 'turn-left', (66.6, -22.2))
    assert goal_in_list(goals, 'turn-right', (46.8, -40.5))
    assert goal_in_list(goals, 'straight-on', (63.1, -45.5))


def test_bendplatz_southeast():
    xodr = "../scenarios/maps/bendplatz.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(135)
    speed = 5
    time = 0
    position = np.array((85.1, -64.6))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 3
    assert goal_in_list(goals, 'turn-left', (46.7, -40.7))
    assert goal_in_list(goals, 'turn-right', (66.8, -22.3))
    assert goal_in_list(goals, 'straight-on', (49.2, -20.9))


def test_frankenberg_northwest():
    xodr = "../scenarios/maps/frankenberg.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(-45)
    speed = 5
    time = 0
    position = np.array((33.6, -10.))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 3
    assert goal_in_list(goals, 'turn-left', (57.9, -30.2))
    assert goal_in_list(goals, 'turn-right', (38.8, -30.6))
    assert goal_in_list(goals, 'straight-on', (45.2, -35.1))


def test_frankenberg_southwest():
    xodr = "../scenarios/maps/frankenberg.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(20)
    speed = 5
    time = 0
    position = np.array((34.28, -34.88))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 3
    assert goal_in_list(goals, 'turn-left', (49.6, -22.9))
    assert goal_in_list(goals, 'turn-right', (45.6, -35.))
    assert goal_in_list(goals, 'straight-on', (57.9, -30.4))


def test_round_north():
    xodr = "../scenarios/maps/round.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(-10)
    speed = 5
    time = 0
    position = np.array((48.3, -43.4))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 4
    assert goal_in_list(goals, 'exit-roundabout', (97.5, -23.0))
    assert goal_in_list(goals, 'exit-roundabout', (108.5, -59.5))
    assert goal_in_list(goals, 'exit-roundabout', (65.8, -70.6))
    assert goal_in_list(goals, 'exit-roundabout', (53.3, -35.9))


def test_round_south():
    xodr = "../scenarios/maps/round.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(-10)
    speed = 5
    time = 0
    position = np.array((48.3, -43.4))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 4
    assert goal_in_list(goals, 'exit-roundabout', (97.5, -23.0))
    assert goal_in_list(goals, 'exit-roundabout', (108.5, -59.5))
    assert goal_in_list(goals, 'exit-roundabout', (65.8, -70.6))
    assert goal_in_list(goals, 'exit-roundabout', (53.3, -35.9))


def test_town01_mainroad():
    xodr = "../scenarios/maps/town01.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(90)
    speed = 5
    time = 0
    position = np.array((92.5, -211.4))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 2
    assert goal_in_list(goals, 'turn-right', (101.4, -199.))
    assert goal_in_list(goals, 'straight-on', (92.3, -186.))


def test_town01_sideroad():
    xodr = "../scenarios/maps/town01.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(180)
    speed = 5
    time = 0
    position = np.array((102.8, -129.6))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    trajectory = VelocityTrajectory.from_agent_state(state)
    goal_generator = GoalGenerator()
    goals = goal_generator.generate(scenario_map, trajectory)

    assert len(goals) == 2
    assert goal_in_list(goals, 'turn-right', (92.4, -119.9))
    assert goal_in_list(goals, 'turn-left', (88.3, -141.9))


def test_town01_view_radius():
    xodr = "../scenarios/maps/town01.xodr"
    scenario_map = Map.parse_from_opendrive(xodr)
    heading = np.deg2rad(0)
    speed = 5
    time = 0
    position = np.array((106.4, -199.2))
    velocity = speed * np.array((np.cos(heading), np.sin(heading)))
    acceleration = np.array((0, 0))

    state = AgentState(time, position, velocity, acceleration, heading)
    goal_generator = GoalGenerator()
    trajectory = VelocityTrajectory.from_agent_state(state)
    goals = goal_generator.generate(scenario_map, trajectory, Circle(position, 50.))

    assert len(goals) == 1
    assert goal_in_list(goals, 'straight-on', (325.7, -199.2))
