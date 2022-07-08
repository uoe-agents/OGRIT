import math
import pickle
import json
from copy import deepcopy
from typing import List

import numpy as np
from itertools import combinations

from ogrit.occlusion_detection.occlusion_line import OcclusionLine as Line
from ogrit.core.data_processing import get_episode_frames
from ogrit.core.base import get_scenarios_dir, get_occlusions_dir
import ogrit.occlusion_detection.visualisation_tools as debug

import igp2 as ip
from igp2.data.scenario import InDScenario, ScenarioConfig
from igp2.opendrive.map import Map


from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import unary_union

# After how many meters can't the vehicle see anything
OCCLUSION_RADIUS = 100


class OcclusionDetector2D:

    def __init__(self, scenario_name: str, episode_idx: int, debug: bool = False, debug_steps: bool = False):
        self.scenario_name = scenario_name
        self.episode_idx = episode_idx
        self.scenario_map = Map.parse_from_opendrive(get_scenarios_dir() + f"/maps/{self.scenario_name}.xodr")
        self.scenario_config = ScenarioConfig.load(get_scenarios_dir() + f"/configs/{self.scenario_name}.json")
        self.scenario = InDScenario(self.scenario_config)
        self.episode = self.scenario.load_episode(episode_idx)
        self.buildings = self.scenario_config.buildings

        # "Debug" mode takes precedence over "debug_steps"
        self.debug = debug
        self.debug_steps = debug_steps if debug_steps and not self.debug else False
        self.occlusion_lines = []

    def extract_occlusions(self, save_format="p"):
        """
        Args:
            save_format: enter "json" to store the occlusion data in json format, or leave empty to store them into
                         pickle
        """

        # episode_frames contains for each time step the list of frames for all vehicles alive that moment
        episode_frames = get_episode_frames(self.episode, exclude_parked_cars=False, exclude_bicycles=True)

        all_occlusion_data = {}

        for frame_id, frame in enumerate(episode_frames):
            print(f"Starting frame {frame_id}/{len(episode_frames) - 1}")

            all_occlusion_data[frame_id] = self.get_occlusions_frame(frame)

        if save_format == "json":
            all_occlusion_data = self._to_json(all_occlusion_data)

        self._save_occlusions(all_occlusion_data, save_format)

    def _to_json(self, occlusions_data):
        """
        Convert the occlusions into a format that can be stored into json. E.g., store the occluded areas by their
        boundaries rather than (Multi)Polygons.
        """
        occlusions_data_json = deepcopy(occlusions_data)

        for frame_id, frame_occlusions in occlusions_data.items():
            for ego_id, ego_occlusions in frame_occlusions.items():
                for road_id, road_occlusions in ego_occlusions.items():
                    for lane_id, lane_occlusions in road_occlusions.items():

                        occlusions_data_json[frame_id][ego_id][road_id][lane_id] = []
                        road_occlusions_data = occlusions_data_json[frame_id][ego_id][road_id][lane_id]

                        if isinstance(lane_occlusions, Polygon):
                            road_occlusions_data.append(self.convert_to_list(lane_occlusions.exterior.xy))
                        elif isinstance(lane_occlusions, MultiPoint):
                            for occlusion in lane_occlusions.geoms:
                                road_occlusions_data.append(self.convert_to_list(occlusion.exterior.xy))
        return occlusions_data_json

    @staticmethod
    def convert_to_list(coordinates: List[np.array]):
        x, y = coordinates
        return [list(x), list(y)]

    def _save_occlusions(self, data_to_store, save_format):

        occlusions_file_name = get_occlusions_dir() + f"/{self.scenario_name}_e{self.episode_idx}.{save_format}"

        if save_format == "p":
            with open(occlusions_file_name, 'wb') as file:
                pickle.dump(data_to_store, file)
        elif save_format == "json":
            with open(occlusions_file_name, 'w') as file:
                json.dump(data_to_store, file)

    def get_occlusions_frame(self, frame):
        frame_occlusions = {}
        vehicles_in_frame = [(vehicle_id, frame.get(vehicle_id)) for vehicle_id in frame.keys()]

        # Get the boundaries of each of the vehicles.
        vehicles_in_frame_boxes = [self.get_box(vehicle) for _, vehicle in vehicles_in_frame]

        # Use each of the vehicles in the frame as ego vehicles in turn.
        for ego_idx, (ego_id, ego_vehicle) in enumerate(vehicles_in_frame):

            # We only want to compute the occlusions for non-parked vehicles.
            if self.episode.agents[ego_id].parked():
                continue

            ego_position = ego_vehicle.position
            other_vehicles_boxes = [vehicle_box for i, vehicle_box in enumerate(vehicles_in_frame_boxes)
                                    if i != ego_idx]

            # Sort the obstacles based on their distance to the ego. Closer objects may hide other objects, which we
            # then don't need to consider.
            obstacles = other_vehicles_boxes
            obstacles.sort(key=lambda box: math.dist(ego_position, box.center))

            obstacles = self.buildings + [list(box.boundary) for box in obstacles]

            ego_vehicle_boundary = list(vehicles_in_frame_boxes[ego_idx].boundary)

            # Get for that ego vehicle, what areas of each lane are occluded.
            ego_occluded_lanes = self.get_occlusions_ego_by_road(ego_position, obstacles, ego_vehicle_boundary)

            if self.debug:
                debug.plot_map(self.scenario_map, self.scenario_config, frame=frame,
                               obstacles=obstacles+[ego_vehicle_boundary])
                debug.plot_occlusions(ego_position, self.occlusion_lines, ego_occluded_lanes,
                                      non_visible_areas=self.non_visible_areas)
                self.occlusion_lines = []
                debug.show_plot()

            frame_occlusions[ego_id] = ego_occluded_lanes
        return frame_occlusions

    def get_occlusions_ego_by_road(self, ego_position, obstacles, ego_vehicle_boundary):
        """
        Get all the occlusions inside each possible lane in the map.

        Args:
            ego_position:         position of the ego vehicle
            obstacles:            list of boundaries of the obstacles that could create occlusions
            ego_vehicle_boundary: boundary of the ego vehicle. Used only when plotting the frame

        Returns:
             A dictionary with the road id as key and another dictionary as value. The latter dictionary has the
             road's lanes id as key and a Multipolygon as value to represent the occluded areas.

        """
        occlusions_by_roads = {k: {} for k in self.scenario_map.roads.keys()}

        # First find all the occluded areas.
        occluded_areas = self.get_occlusions_ego(ego_position, obstacles, ego_vehicle_boundary)

        # Add to the occlusions everything that is more than OCCLUSION_RADIUS meters away.
        non_visible_areas = self._get_occlusions_far_away(ego_position)
        occluded_areas = unary_union([occluded_areas, non_visible_areas])

        # Find what areas in each lane is occluded.
        for road in self.scenario_map.roads.values():
            road_occlusions = {}

            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    if lane.id == 0 or lane.type != "driving":
                        continue

                    intersection = lane.boundary.buffer(0).intersection(occluded_areas)
                    if intersection.is_empty:
                        road_occlusions[lane.id] = None
                        continue

                    road_occlusions[lane.id] = intersection
            occlusions_by_roads[road.id] = road_occlusions

        return occlusions_by_roads

    def _get_occlusions_far_away(self, ego_position):
        """
        Return the area that is OCCLUSIONS_RADIUS meters away from the ego_position. This are is what the ego cannot
        see even if there are no occlusions.
        """
        m = self.scenario_map

        # Get the areas in the map that are more than OCCLUSION_RADIUS meters away from the ego.
        entire_area = Polygon([(m.east, m.north), (m.west, m.north), (m.west, m.south), (m.east, m.south)])
        visible_area = Point(ego_position).buffer(OCCLUSION_RADIUS)

        non_visible_areas = entire_area.difference(visible_area)

        if self.debug:
            self.non_visible_areas = non_visible_areas
        return non_visible_areas

    def get_occlusions_ego(self, ego_position, obstacles, ego_vehicle_boundary):
        """
        Get all the areas (as a Polygon or Multipolygon) that the obstacles occlude from the point of view of
        the ego vehicle.
        """
        occlusions_ego_list = []
        occlusions_ego = Polygon()

        for u in obstacles:

            if occlusions_ego.covers(MultiPoint(u)):
                # The obstacle is already fully occluded by another obstacle.
                continue

            l1, l2 = self.get_occlusion_lines(ego_position, u)

            # When the obstacle is only a point, it doesn't create occlusions
            if l1 is None:
                continue

            # Using the notation in the OGRIT paper.
            v1 = l1.points[1]
            v2 = l2.points[1]

            v3 = l1.get_extended_point(2*OCCLUSION_RADIUS - l1.length, v1)
            v4 = l2.get_extended_point(2*OCCLUSION_RADIUS - l2.length, v2)

            if self.debug or self.debug_steps:
                self.occlusion_lines.append([(v1, v3), (v2, v4)])

            if self.debug_steps:
                debug.plot_map(self.scenario_map, self.scenario_config, obstacles=obstacles+[ego_vehicle_boundary])
                debug.plot_occlusions(ego_position, self.occlusion_lines)
                self.occlusion_lines = []
                debug.show_plot()

            # Find the area that is occluded by obstacle u -- that define by vertices v1, v2, v3, v4.
            occlusions_ego_list.append(Polygon([v1, v2, v4, v3]))
            occlusions_ego = unary_union([geom if geom.is_valid else geom.buffer(0) for geom in occlusions_ego_list])
        return occlusions_ego

    @staticmethod
    def get_occlusion_lines(ego_position, obstacle_box):
        """
        Get the line segments l1 and l2 from the centre of the ego vehicle to vertices v1 and v2 of obstacle_box,
        such that l1 and l2 yield the greatest angle between them
        """

        # Get the endpoints of the lines from the center of the ego to every vertex v of the obstacle.
        lines = [Line(ego_position, vertex) for vertex in obstacle_box]

        l1 = l2 = None
        max_alpha = 0

        for line1, line2 in combinations(lines, 2):
            angle = line1.angle_between(line2)

            if angle > max_alpha:
                max_alpha = angle
                l1, l2 = line1, line2
        return l1, l2

    @staticmethod
    def get_box(vehicle):
        """
        Get the boundaries of the vehicle.
        """
        return ip.Box(np.array([vehicle.position[0],
                                vehicle.position[1]]),
                      vehicle.metadata.length,
                      vehicle.metadata.width,
                      vehicle.heading)

