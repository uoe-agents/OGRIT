import numpy as np
import pytest
from igp2 import AgentState, VelocityTrajectory
from igp2.goal import PointGoal, Goal
from igp2.opendrive.map import Map

from ogrit.core.feature_extraction import FeatureExtractor
from ogrit.core.goal_generator import TypedGoal


def get_feature_extractor():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/heckstrasse.xodr")
    return FeatureExtractor(scenario_map)


def test_angle_in_lane_straight():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/heckstrasse.xodr")
    state = AgentState(time=0,
                       position=np.array((28.9, -21.5)),
                       velocity=np.array((0, 0)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )
    lane = scenario_map.get_lane(1, 2, 0)
    assert FeatureExtractor.angle_in_lane(state, lane) == pytest.approx(0, abs=0.2)


def test_angle_in_lane_curved():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/heckstrasse.xodr")
    state = AgentState(time=0,
                       position=np.array((28.9, -21.5)),
                       velocity=np.array((0, 0)),
                       acceleration=np.array((0, 0)),
                       heading=0
                       )
    lane = scenario_map.get_lane(1, 2, 0)
    assert FeatureExtractor.angle_in_lane(state, lane) == pytest.approx(np.pi/4, abs=0.2)


def test_in_correct_lane():
    feature_extractor = get_feature_extractor()
    scenario_map = feature_extractor.scenario_map
    lane_path = [scenario_map.get_lane(1, 2, 0),
                 scenario_map.get_lane(7, -1, 0)]
    assert feature_extractor.in_correct_lane(lane_path)


def test_not_in_correct_lane():
    feature_extractor = get_feature_extractor()
    scenario_map = feature_extractor.scenario_map
    lane_path = [scenario_map.get_lane(1, 1, 0),
                 scenario_map.get_lane(1, 2, 0),
                 scenario_map.get_lane(7, -1, 0)]
    assert not feature_extractor.in_correct_lane(lane_path)


def test_path_to_goal_length_same_lane():
    feature_extractor = get_feature_extractor()
    state = AgentState(time=0,
                       position=np.array((-7.9, 5.6)),
                       velocity=np.array((0, 0)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )
    lane = feature_extractor.scenario_map.get_lane(1, 2, 0)
    path = [lane]
    goal = TypedGoal('straight-on', PointGoal(np.array((30.7, -22.9)), 1.5), path)

    assert feature_extractor.path_to_goal_length(state, goal, path) == pytest.approx(47.98, abs=1)


def test_path_to_goal_length_different_lane():
    feature_extractor = get_feature_extractor()
    state = AgentState(time=0,
                       position=np.array((-7.9, 5.6)),
                       velocity=np.array((0, 0)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )
    path = [feature_extractor.scenario_map.get_lane(1, 2, 0),
            feature_extractor.scenario_map.get_lane(7, -1, 0)]
    goal = TypedGoal('straight-on', PointGoal(np.array((62.0, -47.4)), 1.5), path)

    assert feature_extractor.path_to_goal_length(state, goal, path) == pytest.approx(87.7, abs=1)


def test_path_to_goal_length_lane_change():
    feature_extractor = get_feature_extractor()
    state = AgentState(time=0,
                       position=np.array((12.2, -4.3)),
                       velocity=np.array((0, 0)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )
    path = [feature_extractor.scenario_map.get_lane(1, 1, 0),
            feature_extractor.scenario_map.get_lane(1, 2, 0),
            feature_extractor.scenario_map.get_lane(7, -1, 0)]
    goal = TypedGoal('straight-on', PointGoal(np.array((62.0, -47.4)), 1.5), path)

    assert feature_extractor.path_to_goal_length(state, goal, path) == pytest.approx(65.8, abs=1)


def test_vehicle_in_front_no_vehicles():
    feature_extractor = get_feature_extractor()
    state = AgentState(time=0,
                       position=np.array((-7.9, 5.6)),
                       velocity=np.array((0, 0)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )
    path = [feature_extractor.scenario_map.get_lane(1, 2, 0),
            feature_extractor.scenario_map.get_lane(7, -1, 0)]
    frame = {0: state}
    vehicle_in_front = feature_extractor.vehicle_in_front(0, path, frame)
    assert vehicle_in_front == (None, np.inf)


def test_vehicle_in_front():
    feature_extractor = get_feature_extractor()
    state0 = AgentState(time=0,
                       position=np.array((-6.5, 4.1)),
                       velocity=np.array((0, 0)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )

    state1 = AgentState(time=0,
                       position=np.array((14.0, -10.8)),
                       velocity=np.array((0, 0)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )
    path = [feature_extractor.scenario_map.get_lane(1, 2, 0),
            feature_extractor.scenario_map.get_lane(7, -1, 0)]
    frame = {0: state0, 1: state1}
    agent_id, dist = feature_extractor.vehicle_in_front(0, path, frame)
    assert agent_id == 1
    assert dist == pytest.approx(25.3, 1)


def test_vehicle_in_front_behind():
    feature_extractor = get_feature_extractor()
    state0 = AgentState(time=0,
                        position=np.array((14.0, -10.8)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=-np.pi / 4
                        )

    state1 = AgentState(time=0,
                        position=np.array((-6.5, 4.1)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=-np.pi / 4
                        )
    path = [feature_extractor.scenario_map.get_lane(1, 2, 0),
            feature_extractor.scenario_map.get_lane(7, -1, 0)]
    frame = {0: state0, 1: state1}
    agent_id, dist = feature_extractor.vehicle_in_front(0, path, frame)
    assert agent_id is None
    assert dist == np.inf


def test_vehicle_in_front_with_lane_change():
    feature_extractor = get_feature_extractor()
    state0 = AgentState(time=0,
                        position=np.array((19.61, -13.96)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=-np.pi / 4
                        )

    state1 = AgentState(time=0,
                        position=np.array((12.25, -5.08)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=-np.pi / 4
                        )
    path = [feature_extractor.scenario_map.get_lane(1, 2, 0),
            feature_extractor.scenario_map.get_lane(1, 1, 0),
            feature_extractor.scenario_map.get_lane(5, -1, 0)]
    frame = {0: state0, 1: state1}
    agent_id, dist = feature_extractor.vehicle_in_front(0, path, frame)
    assert agent_id is None
    assert dist == np.inf


def test_oncoming_vehicle_none():
    feature_extractor = get_feature_extractor()
    state = AgentState(time=0,
                       position=np.array((-7.9, 5.6)),
                       velocity=np.array((0, 0)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )
    path = [feature_extractor.scenario_map.get_lane(1, 2, 0),
            feature_extractor.scenario_map.get_lane(7, -1, 0)]
    frame = {0: state}
    vehicle_in_front = feature_extractor.oncoming_vehicle(0, path, frame)
    assert vehicle_in_front == (None, 100)


def test_oncoming_vehicle():
    feature_extractor = get_feature_extractor()
    state0 = AgentState(time=0,
                       position=np.array((18.0, -8.7)),
                       velocity=np.array((7.07, -7.07)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )

    state1 = AgentState(time=0,
                       position=np.array((68.5, -42.7)),
                       velocity=np.array((-7.07, 7.07)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )
    path = [feature_extractor.scenario_map.get_lane(1, 2, 0),
            feature_extractor.scenario_map.get_lane(5, -1, 0)]
    frame = {0: state0, 1: state1}
    agent_id, dist = feature_extractor.oncoming_vehicle(0, path, frame)
    assert agent_id == 1
    assert dist == pytest.approx(32.4, 1)


def test_road_heading():
    feature_extractor = get_feature_extractor()
    scenario_map = feature_extractor.scenario_map
    lane_path = [scenario_map.get_lane(1, 1, 0),
                 scenario_map.get_lane(5, -1, 0)]
    road_heading = feature_extractor.road_heading(lane_path)
    assert road_heading == pytest.approx(np.pi * 0.3, np.pi * 0.01)


def test_path_to_lane():
    feature_extractor = get_feature_extractor()
    scenario_map = feature_extractor.scenario_map
    lane_path = [scenario_map.get_lane(1, 1, 0),
                 scenario_map.get_lane(1, 2, 0),
                 scenario_map.get_lane(7, -1, 0)]
    path_to_lane = feature_extractor.path_to_lane(lane_path[0], lane_path[-1])
    assert path_to_lane == lane_path


def test_exit_number_1():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/neuweiler.xodr")
    feature_extractor = FeatureExtractor(scenario_map)
    state = AgentState(time=0,
                       position=np.array((71.9, -76.1)),
                       velocity=np.array((0., 0.)),
                       acceleration=np.array((0, 0)),
                       heading=np.pi * 3/8
                       )
    lane_path = [scenario_map.get_lane(18, -1, 0)]
    exit_number = feature_extractor.exit_number(state, lane_path)
    assert exit_number == 1


def test_exit_number_3():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/neuweiler.xodr")
    feature_extractor = FeatureExtractor(scenario_map)
    state = AgentState(time=0,
                       position=np.array((71.9, -76.1)),
                       velocity=np.array((0., 0.)),
                       acceleration=np.array((0, 0)),
                       heading=np.pi * 3/8
                       )
    lane_path = [scenario_map.get_lane(11, -1, 0)]
    exit_number = feature_extractor.exit_number(state, lane_path)
    assert exit_number == 3


def test_exit_number_0():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/neuweiler.xodr")
    feature_extractor = FeatureExtractor(scenario_map)
    state = AgentState(time=0,
                       position=np.array((88.5, -62.9)),
                       velocity=np.array((0., 0.)),
                       acceleration=np.array((0, 0)),
                       heading=np.pi * 1/4
                       )
    lane_path = [scenario_map.get_lane(11, -1, 0)]
    exit_number = feature_extractor.exit_number(state, lane_path)
    assert exit_number == 0


def test_exit_number_no_roundabout():
    feature_extractor = get_feature_extractor()
    state = AgentState(time=0,
                       position=np.array((-7.9, 5.6)),
                       velocity=np.array((0, 0)),
                       acceleration=np.array((0, 0)),
                       heading=-np.pi/4
                       )
    lane = feature_extractor.scenario_map.get_lane(1, 2, 0)
    lane_path = [lane]
    exit_number = feature_extractor.exit_number(state, lane_path)
    assert exit_number == 0


def test_in_correct_lane_neukoellnerstrasse():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/neukoellnerstrasse.xodr")
    feature_extractor = FeatureExtractor(scenario_map)
    lane_path = [scenario_map.get_lane(4, 1, 0),
                 scenario_map.get_lane(5, 1, 0),
                 scenario_map.get_lane(11, -1, 0)]
    assert feature_extractor.in_correct_lane(lane_path)


def test_not_in_correct_lane_neukoellnerstrasse():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/neukoellnerstrasse.xodr")
    feature_extractor = FeatureExtractor(scenario_map)
    lane_path = [scenario_map.get_lane(4, 2, 0),
                 scenario_map.get_lane(5, -2, 0),
                 scenario_map.get_lane(5, -1, 0),
                 scenario_map.get_lane(9, -1, 0)]
    assert not feature_extractor.in_correct_lane(lane_path)


def test_exit_number_rdb5():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/rdb5.xodr")
    feature_extractor = FeatureExtractor(scenario_map)
    state = AgentState(time=0,
                       position=np.array((16.1, -29.6)),
                       velocity=np.array((0., 0.)),
                       acceleration=np.array((0, 0)),
                       heading=np.pi * 5/8
                       )
    lane_path = [scenario_map.get_lane(3, -1, 0),
                 scenario_map.get_lane(8, -1, 0),
                 scenario_map.get_lane(178, -1, 0),
                 scenario_map.get_lane(178, -4, 1),
                 scenario_map.get_lane(178, -1, 2)]

    exit_number = feature_extractor.exit_number(state, lane_path)
    assert exit_number == 2


def test_slip_road_true():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/rdb6.xodr")
    feature_extractor = FeatureExtractor(scenario_map, 'rdb6')
    slip_road = feature_extractor.slip_road(1, TypedGoal('roundabout-exit',
                                    PointGoal(np.array((15.7, -11.5)), 1.5), []))
    assert slip_road


def test_slip_road_false():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/rdb6.xodr")
    feature_extractor = FeatureExtractor(scenario_map, 'rdb6')
    slip_road = feature_extractor.slip_road(3, TypedGoal('roundabout-exit',
                                    PointGoal(np.array((15.7, -11.5)), 1.5), []))
    assert not slip_road


def test_slip_road_false_2():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/rdb6.xodr")
    feature_extractor = FeatureExtractor(scenario_map, 'rdb6')
    slip_road = feature_extractor.slip_road(0, TypedGoal('roundabout-exit',
                                    PointGoal(np.array((-19.9, 17.2)), 1.5), []))
    assert not slip_road


def test_uturn_true():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/neuweiler.xodr")
    feature_extractor = FeatureExtractor(scenario_map, 'neuweiler')
    is_uturn = feature_extractor.is_roundabout_uturn(4)
    assert is_uturn


def test_uturn_false():
    scenario_map = Map.parse_from_opendrive(f"../scenarios/maps/neuweiler.xodr")
    feature_extractor = FeatureExtractor(scenario_map, 'neuweiler')
    is_uturn = feature_extractor.is_roundabout_uturn(2)
    assert not is_uturn
