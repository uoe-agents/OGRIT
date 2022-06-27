import numpy as np
from igp2 import AgentState, plot_map
from igp2.data import ScenarioConfig, InDScenario
from igp2.opendrive.map import Map

import matplotlib.pyplot as plt
from shapely.ops import unary_union

from ogrit.core.data_processing import get_episode_frames
from ogrit.core.feature_extraction import FeatureExtractor
from ogrit.occlusion_detection.occlusion_detection_geometry import OcclusionDetector2D
from ogrit.core.base import get_base_dir


def get_feature_extractor(episode_idx=1, scenario_name="bendplatz"):
    scenario_map = Map.parse_from_opendrive(get_base_dir() + f"/scenarios/maps/{scenario_name}.xodr")
    return FeatureExtractor(scenario_map, scenario_name, episode_idx)


def plot_occlusion(frame_id=153, episode_idx=1, *frame, plot_occlusions=True, all_vehicles=False,
                   scenario_name="bendplatz"):
    feature_extractor = get_feature_extractor(episode_idx=episode_idx, scenario_name=scenario_name)
    occlusions = feature_extractor.occlusions[frame_id]

    scenario_config = ScenarioConfig.load(get_base_dir() + f"/scenarios/configs/{scenario_name}.json")
    scenario = InDScenario(scenario_config)
    episode = scenario.load_episode(feature_extractor.episode_idx)

    # Take a step every 25 recorded frames (1s)
    # episode_frames contain for each second the list of frames for all vehicles alive that moment
    episode_frames = get_episode_frames(episode, exclude_parked_cars=False, exclude_bicycles=True, step=25)

    ego_id = list(occlusions.keys())[0]
    ego_occlusions = occlusions[ego_id]

    ego = episode_frames[frame_id][ego_id]

    plot_map(feature_extractor.scenario_map, scenario_config=scenario_config, plot_buildings=True)

    if plot_occlusions:
        lane_occlusions_all = []

        for road_occlusions in ego_occlusions:
            for lane_occlusions in ego_occlusions[road_occlusions]:
                lane_occlusion = ego_occlusions[road_occlusions][lane_occlusions]

                if lane_occlusion is not None:
                    lane_occlusions_all.append(lane_occlusion)
        OcclusionDetector2D.plot_area_from_list(lane_occlusions_all, color="r", alpha=0.5)

    if all_vehicles:
        for aid, state in episode_frames[frame_id].items():
            plt.text(*state.position, aid)
            plt.plot(*list(zip(*OcclusionDetector2D.get_box(state).boundary)), color="black")

    if frame:
        for aid, state in frame[0].items():
            plt.text(*state.position, aid)
            plt.plot(*list(zip(*OcclusionDetector2D.get_box(state).boundary)))

    plt.plot(*list(zip(*OcclusionDetector2D.get_box(ego).boundary)))


def find_lane_at(point, scenario_name="bendplatz"):
    scenario_map = Map.parse_from_opendrive(get_base_dir() + f"/scenarios/maps/{scenario_name}.xodr")
    lanes = scenario_map.lanes_at(point)

    for lane in lanes:
        plot_map(scenario_map)
        lane = scenario_map.get_lane(lane.parent_road.id, lane.id)
        plt.plot(*list(zip(*[x for x in lane.midline.coords])))
        plt.show()


def get_occlusions_and_ego(frame=153, episode_idx=1):
    feature_extractor = get_feature_extractor(episode_idx)
    occlusions = feature_extractor.occlusions[frame]

    ego_id = list(occlusions.keys())[0]
    ego_occlusions = occlusions[ego_id]

    occlusions = []
    for road_occlusions in ego_occlusions:
        for lane_occlusions in ego_occlusions[road_occlusions]:
            lane_occlusion = ego_occlusions[road_occlusions][lane_occlusions]

            if lane_occlusion is not None:
                occlusions.append(lane_occlusion)
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
    # plot_occlusion(153, 1, frame)
    oncoming_vehicle_id, oncoming_vehicle_dist = mfe.oncoming_vehicle(0, lane_path, frame)
    missing = mfe.is_oncoming_vehicle_missing(oncoming_vehicle_dist, lane_path, occlusions)
    plt.show()

    assert missing


def set_up_frame_ep3_frame100(third_agent_position, third_agent_heading):
    """
    The third agent is the possible oncoming vehicle.
    State 1 is the target vehicle.
    """
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

    target_id = 0
    frame = {target_id: state0, 1: state1, ego_id: state_ego}
    # plot_occlusion(frame_id, episode_idx, frame)
    oncoming_vehicle_id, oncoming_vehicle_dist = mfe.oncoming_vehicle(target_id, lane_path, frame)
    missing = mfe.is_oncoming_vehicle_missing(oncoming_vehicle_dist, lane_path, occlusions)
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

    missing = set_up_frame_ep3_frame100((55.75, -37.73), 180)
    assert not missing


# Tests for missing vehicle ahead.
def test_the_vehicle_in_front_is_hidden():
    """
    State1 is the possible vehicle in front.
    """
    episode_idx = 6
    frame_id = 50

    mfe = get_feature_extractor(episode_idx=episode_idx)

    lane_path = [mfe.scenario_map.get_lane(1, 1, 0)]
    ego_id, occlusions = get_occlusions_and_ego(frame=frame_id, episode_idx=episode_idx)

    state_target = AgentState(time=0,
                              position=np.array((34.58, -56.93)),
                              velocity=np.array((0, 0)),
                              acceleration=np.array((0, 0)),
                              heading=lane_path[0].get_heading_at(45.67, -46.72)
                              )

    state1 = AgentState(time=0,
                        position=np.array((39.90, -52.22)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=np.deg2rad(45)
                        )

    state_ego = AgentState(time=0,
                           position=np.array((34.62, -11.01)),
                           velocity=np.array((0, 0)),
                           acceleration=np.array((0, 0)),
                           heading=np.deg2rad(-45)
                           )
    target_id = 0
    frame = {target_id: state_target, 1: state1, ego_id: state_ego}
    # plot_occlusion(frame_id, episode_idx, frame)
    vehicle_in_front_id, vehicle_in_front_dist = mfe.vehicle_in_front(target_id, lane_path, frame)
    missing = mfe.is_vehicle_in_front_missing(vehicle_in_front_dist, target_id, lane_path, frame, occlusions)
    plt.show()

    assert missing


def test_vehicle_is_behind():
    """
    State1 is the possible vehicle in front.
    """
    episode_idx = 6
    frame_id = 50

    mfe = get_feature_extractor(episode_idx=episode_idx)

    lane_path = [mfe.scenario_map.get_lane(3, -1, 0)]
    ego_id, occlusions = get_occlusions_and_ego(frame=frame_id, episode_idx=episode_idx)

    state_target = AgentState(time=0,
                              position=np.array((76.54, -11.56)),
                              velocity=np.array((0, 0)),
                              acceleration=np.array((0, 0)),
                              heading=lane_path[0].get_heading_at(76.54, -11.56)
                              )

    state1 = AgentState(time=0,
                        position=np.array((68.24, -20.61)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=np.deg2rad(45)
                        )

    state_ego = AgentState(time=0,
                           position=np.array((34.62, -11.01)),
                           velocity=np.array((0, 0)),
                           acceleration=np.array((0, 0)),
                           heading=np.deg2rad(-45)
                           )
    target_id = 0
    frame = {target_id: state_target, 1: state1, ego_id: state_ego}
    # plot_occlusion(frame_id, episode_idx, frame)
    vehicle_in_front_id, vehicle_in_front_dist = mfe.vehicle_in_front(target_id, lane_path, frame)
    missing = mfe.is_vehicle_in_front_missing(vehicle_in_front_dist, target_id, lane_path, frame, occlusions)
    plt.show()

    assert missing


def test_no_vehicle_in_front_2():
    """
    State1 is the possible vehicle in front.
    """
    episode_idx = 6
    frame_id = 50

    mfe = get_feature_extractor(episode_idx=episode_idx)

    lane_path = [mfe.scenario_map.get_lane(3, -1, 0)]
    ego_id, occlusions = get_occlusions_and_ego(frame=frame_id, episode_idx=episode_idx)

    state_target = AgentState(time=0,
                              position=np.array((72.77, -9.44)),
                              velocity=np.array((0, 0)),
                              acceleration=np.array((0, 0)),
                              heading=lane_path[0].get_heading_at(72.77, -9.44)
                              )

    state1 = AgentState(time=0,
                        position=np.array((66.29, -16.77)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=np.deg2rad(45)
                        )

    state_ego = AgentState(time=0,
                           position=np.array((34.62, -11.01)),
                           velocity=np.array((0, 0)),
                           acceleration=np.array((0, 0)),
                           heading=np.deg2rad(-45)
                           )
    target_id = 0
    frame = {target_id: state_target, 1: state1, ego_id: state_ego}
    # plot_occlusion(frame_id, episode_idx, frame)
    vehicle_in_front_id, vehicle_in_front_dist = mfe.vehicle_in_front(target_id, lane_path, frame)
    missing = mfe.is_vehicle_in_front_missing(vehicle_in_front_dist, target_id, lane_path, frame, occlusions)
    plt.show()

    assert not missing

def test_occlusion_far_away():
    episode_idx = 7
    frame_id = 200

    mfe = get_feature_extractor(episode_idx=episode_idx)

    lane_path = [mfe.scenario_map.get_lane(2, 2, 0),
                 mfe.scenario_map.get_lane(10, -1, 0)]
    ego_id, occlusions = get_occlusions_and_ego(frame=frame_id, episode_idx=episode_idx)

    state_target = AgentState(time=0,
                              position=np.array((84.70, -60.43)),
                              velocity=np.array((0, 0)),
                              acceleration=np.array((0, 0)),
                              heading=lane_path[0].get_heading_at(84.70, -60.43)
                              )

    state_ego = AgentState(time=0,
                           position=np.array((73.39, -56.32)),
                           velocity=np.array((0, 0)),
                           acceleration=np.array((0, 0)),
                           heading=np.deg2rad(-45)
                           )
    target_id = 0
    frame = {target_id: state_target, ego_id: state_ego}
    # plot_occlusion(frame_id, episode_idx, frame)
    vehicle_in_front_id, vehicle_in_front_dist = mfe.vehicle_in_front(target_id, lane_path, frame)
    missing = mfe.is_vehicle_in_front_missing(vehicle_in_front_dist, target_id, lane_path, frame, occlusions)
    plt.show()

    assert not missing


def test_occlusion_close_enough():

    episode_idx = 7
    frame_id = 200

    mfe = get_feature_extractor(episode_idx=episode_idx)

    lane_path = [mfe.scenario_map.get_lane(10, -1, 0)]
    ego_id, occlusions = get_occlusions_and_ego(frame=frame_id, episode_idx=episode_idx)

    state_target = AgentState(time=0,
                              position=np.array((61.59, -34.41)),
                              velocity=np.array((0, 0)),
                              acceleration=np.array((0, 0)),
                              heading=lane_path[0].get_heading_at(61.59, -34.41)
                              )

    state_ego = AgentState(time=0,
                           position=np.array((73.39, -56.32)),
                           velocity=np.array((0, 0)),
                           acceleration=np.array((0, 0)),
                           heading=np.deg2rad(-45)
                           )
    target_id = 0
    frame = {target_id: state_target, ego_id: state_ego}
    # plot_occlusion(frame_id, episode_idx, frame)
    vehicle_in_front_id, vehicle_in_front_dist = mfe.vehicle_in_front(target_id, lane_path, frame)
    missing = mfe.is_vehicle_in_front_missing(vehicle_in_front_dist, target_id, lane_path, frame, occlusions)
    plt.show()

    assert missing

def test_occlusion_between_vehicle_in_front():
    """
    State1 is the possible vehicle in front.
    """
    episode_idx = 6
    frame_id = 42

    mfe = get_feature_extractor(episode_idx=episode_idx)

    lane_path = [mfe.scenario_map.get_lane(1, 1, 0),
                 mfe.scenario_map.get_lane(7, -1, 0)]
    ego_id, occlusions = get_occlusions_and_ego(frame=frame_id, episode_idx=episode_idx)

    state_target = AgentState(time=0,
                              position=np.array((33.07, -58.33)),
                              velocity=np.array((0, 0)),
                              acceleration=np.array((0, 0)),
                              heading=lane_path[0].get_heading_at(33.07, -58.33)
                              )

    state1 = AgentState(time=0,
                        position=np.array((43.62, -48.47)),
                        velocity=np.array((0, 0)),
                        acceleration=np.array((0, 0)),
                        heading=np.deg2rad(45)
                        )

    state_ego = AgentState(time=0,
                           position=np.array((73.39, -56.32)),
                           velocity=np.array((0, 0)),
                           acceleration=np.array((0, 0)),
                           heading=np.deg2rad(-45)
                           )
    target_id = 0
    frame = {target_id: state_target, ego_id: state_ego, 1: state1}
    # plot_occlusion(frame_id, episode_idx, frame)
    vehicle_in_front_id, vehicle_in_front_dist = mfe.vehicle_in_front(target_id, lane_path, frame)
    missing = mfe.is_vehicle_in_front_missing(vehicle_in_front_dist, target_id, lane_path, frame, occlusions)
    plt.show()

    assert missing

# find_lane_at((32.7, -59.4))
# plot_occlusion(42, 5, scenario_name="bendplatz")
# plt.show()
