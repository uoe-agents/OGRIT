# todo INSTRUCTIONS: you must run the script from the base directory (...\GRIT-OpenDrive\)

import igp2 as ip
import argparse
import json
from typing import List

from igp2.data.scenario import InDScenario, ScenarioConfig
from igp2.opendrive.map import Map
from grit.core.data_processing import get_episode_frames


import matplotlib.pyplot as plt
import numpy as np
import math

import shapely.geometry
from shapely.ops import unary_union
from shapely.geometry import Polygon
from sympy import Point, Segment # todo: only use shapely -- remove from requirements

from scripts.preprocess_data import iterate_through_scenarios

# How many meters away from the vehicle do we want to detect occlusions.
OCCLUSION_LINE_LENGTH = 50


def parse_args():
    parser = argparse.ArgumentParser(description="""
    This script detects what parts of the lanes are occluded using a bird's eye view of the map 
         """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--debug',
                        help="if set, we plot all the occlusions in a frame for each vehicle."
                             "If --debug_steps is also True, this takes precedence and --debug_steps will be"
                             "deactivated.",
                        action='store_true')

    parser.add_argument('--debug_steps',
                        help="if set, we plot the occlusions created by each obstacle. "
                             "If --debug is set, --debug_steps will be disabled.",
                        action='store_true')

    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
    parser.add_argument('--workers', type=int, help='Number of multiprocessing workers', default=4)

    parsed_config_specification = vars(parser.parse_args())
    return parsed_config_specification


def main(params):

    config = parse_args()
    if config["debug"]:
        config["debug_steps"] = "False"

    scenario_name, episode_idx = params
    print('scenario {} episode {}'.format(scenario_name, episode_idx))

    scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{scenario_name}.xodr")
    scenario_config = ScenarioConfig.load(f"scenarios/configs/{scenario_name}.json")
    scenario = InDScenario(scenario_config)
    episode = scenario.load_episode(episode_idx)
    buildings = scenario_config.buildings

    # Take a step every 25 recorded frames (1s)
    # episode_frames contain for each second the list of frames for all vehicles alive that moment
    episode_frames = get_episode_frames(episode, exclude_parked_cars=False, exclude_bicycles=True, step=25)
    all_occlusion_data = {}

    for frame_id, frame in enumerate(episode_frames):

        print(f"Starting frame {frame_id}/{len(episode_frames)}")
        vehicles_in_frame_ids = list(frame.keys())

        frame_occlusions = []

        for main_vehicle_id in vehicles_in_frame_ids:

            main_vehicle = frame.get(main_vehicle_id)

            # We only want to compute the occlusions for the non-parked vehicles.
            if episode.agents[main_vehicle_id].parked():
                continue

            other_vehicles = [frame.get(vehicle_id) for vehicle_id in vehicles_in_frame_ids if
                              vehicle_id != main_vehicle_id]

            # Store the boundaries of the boxes containing possible obstacles (other vehicles and buildings).
            obstacle_boxes = [list(get_box(vehicle).boundary) for vehicle in other_vehicles] + buildings
            occlusions = {k: {} for k in scenario_map.roads.keys()}

            if config["debug"]:
                plot_map(scenario_map, frame, obstacle_boxes)

            for obstacle_box in obstacle_boxes:

                lines = []
                for vertex in obstacle_box:
                    p = Point(vertex[0], vertex[1])
                    lines.append(Segment(Point(*main_vehicle.position), p))

                max_alpha = 0
                max_l = None
                unvisited_lines = lines.copy()
                for line in lines:
                    unvisited_lines.remove(line)
                    for l in unvisited_lines:
                        angle = line.angle_between(l)

                        if angle > max_alpha:
                            max_alpha = angle
                            max_l = [line, l]

                if max_l is None:  # When the obstacle is only a point, it doesn't create occlusions
                    continue

                line1 = max_l[0]
                line2 = max_l[1]

                p1 = get_extended_point(OCCLUSION_LINE_LENGTH - line1.length, line1.slope, line1.direction, line1.points[1])
                p2 = get_extended_point(OCCLUSION_LINE_LENGTH - line2.length, line2.slope, line2.direction, line2.points[1])

                vertices = [get_shapely_point(line1.points[1]), get_shapely_point(line2.points[1]),
                            get_shapely_point(p2), get_shapely_point(p1)]

                occluded_area = Polygon(vertices)  # todo: reuse line1.points[1]

                # todo: find the vertex that is farthest away from the center of the occluded polygon.
                #  This is used to determine which radius we should use to find the lanes that could be occluded.
                occluded_area_center = occluded_area.centroid
                farthest_distance = max(list(map(lambda x: occluded_area_center.distance(x), vertices)))

                lanes_nearby = scenario_map.lanes_at(occluded_area_center, drivable_only=True, max_distance=farthest_distance)

                if config["debug_steps"]:
                    plot_map(scenario_map, frame, obstacle_boxes)
                    list_intersections = [] # todo:

                for lane in lanes_nearby:

                    intersection = lane.boundary.buffer(0).intersection(occluded_area)

                    if config["debug_steps"] and intersection:
                        list_intersections.append(intersection.exterior.xy)

                    if not intersection:
                        continue

                    road_id = lane.parent_road.id

                    if lane.id not in list(occlusions[road_id].keys()):
                        occlusions[road_id][lane.id] = []

                    occlusions[road_id][lane.id].append(intersection)

                if config["debug_steps"]:
                    plot_occlusions(line1, line2, occluded_area, list_intersections)
                    plt.show()
                if config["debug"]:
                    plot_occlusions(line1, line2, occluded_area, [])

            for road_nr in occlusions.keys():
                for lane_nr in occlusions[road_nr]:

                    unions = unary_union(occlusions[road_nr][lane_nr])

                    if unions.geom_type == "MultiPolygon":
                        unions = list(unions.geoms)
                    else:
                        unions = [unions]

                    occlusions[road_nr][lane_nr] = [convert_to_list(union.exterior.xy) for union in unions]

                    if config["debug"]:
                        plot_occlusions([], [], [], occlusions[road_nr][lane_nr])

            if config["debug"]:
                plt.show()

            frame_occlusions.append({"ego_agent_id": main_vehicle_id,
                                     "occlusions": occlusions})

        all_occlusion_data[frame_id] = frame_occlusions

    json_file_name = f"occlusions/{scenario_name}_e{episode_idx}.json"
    with open(json_file_name, 'w+') as json_file:
        json.dump(all_occlusion_data, json_file, indent=4)

    print('finished scenario {} episode {}'.format(scenario_name, episode_idx))


def convert_to_list(coordinates: List[np.array]):
    x, y = coordinates
    return [list(x), list(y)]


def get_shapely_point(sympy_point):
    return shapely.geometry.Point(sympy_point.x, sympy_point.y)


def get_box(vehicle):
    return ip.Box(np.array([vehicle.position[0],
                            vehicle.position[1]]),
                  vehicle.metadata.length,
                  vehicle.metadata.width,
                  vehicle.heading)


# todo: add description a=endpoint_line1, b=point_in_common/intersection, c=endpoint_line2 -- if using shapely
# todo: use dot product instead
def angle_between(a, b, c):
    # from https://github.com/martinfleis/momepy/blob/9af821dfaf05ab7abdde5e96e12de0e90c4481b4/momepy/utils.py#L112
    ba = [aa - bb for aa, bb in zip(a, b)]
    bc = [cc - bb for cc, bb in zip(c, b)]
    nba = math.sqrt(sum((x ** 2.0 for x in ba)))
    ba = [x / nba for x in ba]
    nbc = math.sqrt(sum((x ** 2.0 for x in bc)))
    bc = [x / nbc for x in bc]
    scal = sum((aa * bb for aa, bb in zip(ba, bc)))
    angle = math.acos(round(scal, 10))
    return angle


# todo: explain arguments and functionality with docs
def get_extended_point(length, slope, direction, point):
    delta_x = math.sqrt(length**2 / (1 + slope**2))
    delta_y = math.sqrt(length**2 - delta_x**2)

    if direction.x < 0:
        delta_x = -delta_x

    if direction.y < 0:
        delta_y = -delta_y

    return point.translate(delta_x, delta_y)


def plot_map(scenario_map, frame, obstacles):
    ip.plot_map(scenario_map, markings=False, midline=False)

    if frame:
        for aid, state in frame.items():
            plt.plot(*state.position, marker="x")
            plt.text(*state.position, aid)

    for obstacle in obstacles:
        # Add the first point also at the end, so we plot a closed contour of the obstacle.
        obstacle.append((obstacle[0]))
        plt.plot(*list(zip(*obstacle)))


def plot_occlusions(line1, line2, occluded_area, lane_occlusions):

    if line1:
        plt.plot(*list(zip(*line1.points)), color='g')
        plt.plot(*list(zip(*line2.points)), color='g')

    if occluded_area:
        plt.plot(*occluded_area.exterior.xy, color="b")

    if lane_occlusions:
        for lane_occlusion in lane_occlusions:
            plt.plot(*lane_occlusion, color="r")


if __name__ == "__main__":

    config = parse_args()
    iterate_through_scenarios(main, config["scenario"], config["workers"])


