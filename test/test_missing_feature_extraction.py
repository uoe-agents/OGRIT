import numpy as np
import pytest
from igp2 import AgentState, VelocityTrajectory, plot_map
from igp2.data import ScenarioConfig, InDScenario
from igp2.goal import PointGoal
from igp2.opendrive.map import Map

import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from grit.core.data_processing import get_episode_frames
from grit.core.feature_extraction import FeatureExtractor
from grit.occlusion_detection.occlusion_detection_geometry import get_box
from grit.core.goal_generator import TypedGoal


scenario_name = "bendplatz"


def get_feature_extractor(episode_idx=1):
    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{scenario_name}.xodr")
    return FeatureExtractor(scenario_map, scenario_name, episode_idx)


def plot_occlusion(frame_id=153, episode_idx=1, *frame, plot_occlusions=False, all_vehicles=False,):
    feature_extractor = get_feature_extractor(episode_idx)
    occlusions = feature_extractor.occlusions

    scenario_config = ScenarioConfig.load(f"scenarios/configs/{scenario_name}.json")
    scenario = InDScenario(scenario_config)
    episode = scenario.load_episode(feature_extractor.episode_idx)

    # Take a step every 25 recorded frames (1s)
    # episode_frames contain for each second the list of frames for all vehicles alive that moment
    episode_frames = get_episode_frames(episode, exclude_parked_cars=False, exclude_bicycles=True, step=25)

    ego_occlusions = occlusions[str(frame_id)][0]["occlusions"]
    ego_id = occlusions[str(frame_id)][0]["ego_agent_id"]

    ego = episode_frames[frame_id][ego_id]

    plot_map(feature_extractor.scenario_map, scenario_config=scenario_config, plot_buildings=True)

    if plot_occlusions:
        for road_occlusions in ego_occlusions:
            for lane_occlusions in ego_occlusions[road_occlusions]:
                for lane_occlusion in ego_occlusions[road_occlusions][lane_occlusions]:
                    plt.plot(*lane_occlusion, color="r")

    if all_vehicles:
        for aid, state in episode_frames[frame_id].items():
            plt.text(*state.position, aid)
            plt.plot(*list(zip(*get_box(state).boundary)), color="black")

    if frame:
        for aid, state in frame[0].items():
            plt.text(*state.position, aid)
            plt.plot(*list(zip(*get_box(state).boundary)))

    plt.plot(*list(zip(*get_box(ego).boundary)))


def find_lane_at(point):
    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{scenario_name}.xodr")
    lanes = scenario_map.lanes_at(point)

    for lane in lanes:
        plot_map(scenario_map)
        lane = scenario_map.get_lane(lane.parent_road.id, lane.id)
        plt.plot(*list(zip(*[x for x in lane.midline.coords])))
        plt.show()


def get_occlusions_and_ego(frame=153, episode_idx=1):
    feature_extractor = get_feature_extractor(episode_idx)
    occlusions = feature_extractor.occlusions

    ego_occlusions = occlusions[str(frame)][0]["occlusions"]
    ego_id = occlusions[str(frame)][0]["ego_agent_id"]

    occlusions = []
    for road_occlusions in ego_occlusions:
        for lane_occlusions in ego_occlusions[road_occlusions]:
            for lane_occlusion in ego_occlusions[road_occlusions][lane_occlusions]:
                occlusions.append(Polygon(list(zip(*lane_occlusion))))
    occlusions = unary_union(occlusions)

    return ego_id, occlusions


def test_occluded_area_no_vehicle_in_oncoming_lanes():
    mfe = get_feature_extractor()

    lane_path = [mfe.scenario_map.get_lane(8, -1, 0)]
    ego_id, occlusions = get_occlusions_and_ego()

    state0 = AgentState(time=0,
                        position=np.array((45.67, -46.72)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=lane_path[0].get_heading_at(45.67, -46.72)
                        )

    state1 = AgentState(time=0,
                        position=np.array((62.88, -20.96)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=np.deg2rad(-120)
                        )

    state_ego = AgentState(time=0,
                           position=np.array((43.88, -44.25)),
                           velocity=np.array((0, 0)),
                           acceleration=np.array((0, 0)),
                           heading=np.deg2rad(45)
                           )

    frame = {ego_id: state_ego, 0: state0, 1: state1}
    plot_occlusion(153, 1, frame)
    missing = mfe.is_oncoming_vehicle_missing(0, lane_path, frame, occlusions)
    plt.show()

    assert missing


def set_up_frame_ep3_frame100(third_agent_position, third_agent_heading):
    episode_idx = 3
    frame_id = 100

    mfe = get_feature_extractor(episode_idx=episode_idx)

    lane_path = [mfe.scenario_map.get_lane(1, 1, 0),
                 mfe.scenario_map.get_lane(9, -1, 0)]
    ego_id, occlusions = get_occlusions_and_ego(frame=frame_id, episode_idx=episode_idx)

    state0 = AgentState(time=0,
                        position=np.array((45.67, -46.72)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=lane_path[0].get_heading_at(45.67, -46.72)
                        )

    state1 = AgentState(time=0,
                        position=np.array(third_agent_position),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=np.deg2rad(third_agent_heading)
                        )

    state_ego = AgentState(time=0,
                           position=np.array((43.88, -44.25)),
                           velocity=np.array((0, 0)),
                           acceleration=np.array((0, 0)),
                           heading=np.deg2rad(45)
                           )

    frame = {0: state0, 1: state1, ego_id: state_ego}
    plot_occlusion(frame_id, episode_idx, frame)
    missing = mfe.is_oncoming_vehicle_missing(ego_id, lane_path, frame, occlusions)
    plt.show()

    return missing


def test_occluded_area_vehicle_in_oncoming_lanes():

    missing = set_up_frame_ep3_frame100((62.88, -20.96), -110)
    assert missing


def test_occluded_area_vehicle_in_oncoming_lanes_2():

    missing = set_up_frame_ep3_frame100((60.12, -33.10), 140)
    assert missing


def test_occluded_area_vehicle_in_oncoming_lanes_3():

    missing = set_up_frame_ep3_frame100((49.12, -30.13), -45)
    assert missing


def test_occluded_area_vehicle_in_oncoming_lanes_4():

    missing = set_up_frame_ep3_frame100((53.81, -38.10), 170)
    assert not missing


def test_occluded_area_vehicle_in_oncoming_lanes_5():

    missing = set_up_frame_ep3_frame100((56.46, -38.11), -45)
    assert missing

def test_occluded_area_vehicle_in_oncoming_lanes_6():

    missing = set_up_frame_ep3_frame100((55.75, -37.73), 30)
    assert not missing

