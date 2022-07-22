import math
import pickle
import json
from typing import List

import numpy as np
from itertools import combinations

from ogrit.occlusion_detection.occlusion_line import OcclusionLine as Line
from ogrit.core.data_processing import get_episode_frames
from ogrit.core.base import get_scenarios_dir, get_occlusions_dir
import ogrit.occlusion_detection.visualisation_tools as util

from igp2.data.scenario import InDScenario, ScenarioConfig
from igp2.opendrive.map import Map


from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from shapely.ops import unary_union

# After how many meters can't the vehicle see anything
OCCLUSION_RADIUS = 100


class OcclusionDetector2D:

    def __init__(self, scenario_name: str, episode_idx: int, debug: bool = False):
        self.scenario_name = scenario_name
        self.episode_idx = episode_idx
        self.scenario_map = Map.parse_from_opendrive(get_scenarios_dir() + f"/maps/{self.scenario_name}.xodr")
        self.scenario_config = ScenarioConfig.load(get_scenarios_dir() + f"/configs/{self.scenario_name}.json")
        self.scenario = InDScenario(self.scenario_config)
        self.episode = self.scenario.load_episode(episode_idx)
        self.buildings = self.scenario_config.buildings
        self.save_format = "p"  # By default, save the occlusions in a pickle file.

        # Whether we want to plot the occlusions w.r.t. each vehicle.
        self.debug = debug
        self.occlusion_lines = []

    def extract_occlusions(self, save_format="p"):
        """
        Args:
            save_format: enter "json" to store the occlusion data in json format, or leave empty to store them into
                         pickle
        """

        self.save_format = save_format

        # episode_frames contains for each time step the list of frames for all vehicles alive that moment
        episode_frames = get_episode_frames(self.episode, exclude_parked_cars=False, exclude_bicycles=True)

        all_occlusion_data = {}

        for frame_id, frame in enumerate(episode_frames):
            print(f"Starting frame {frame_id}/{len(episode_frames) - 1}")

            all_occlusion_data[frame_id] = self.get_occlusions_frame(frame)
        self._save_occlusions(all_occlusion_data)

    def _get_format(self, polygon: MultiPolygon):
        """
        Convert the given (Multi)Polygon in the format in which we want to save the occlusions, as given by
        the value stored in self.save_format.
        """

        def convert_to_list(coordinates: List[np.array]):
            x, y = coordinates
            return [list(x), list(y)]

        if self.save_format == "p":
            return polygon if not polygon.is_empty else None

        elif self.save_format == "json":
            boundaries = []
            if not polygon.is_empty:
                if isinstance(polygon, Polygon):
                    boundaries.append(convert_to_list(polygon.exterior.xy))
                elif isinstance(polygon, MultiPolygon):
                    for polyg in polygon.geoms:
                        boundaries.append(convert_to_list(polyg.exterior.xy))
            return boundaries
        else:
            raise ValueError('You can only store the occlusions either in pickle (save_format="p") " \
                             "or JSON (save_format="json"')

    def _save_occlusions(self, data_to_store):

        occlusions_file_name = get_occlusions_dir() + f"/{self.scenario_name}_e{self.episode_idx}.{self.save_format}"

        if self.save_format == "p":
            with open(occlusions_file_name, 'wb') as file:
                pickle.dump(data_to_store, file)
        elif self.save_format == "json":
            with open(occlusions_file_name, 'w') as file:
                json.dump(data_to_store, file)

    def get_occlusions_frame(self, frame):
        frame_occlusions = {}
        vehicles_in_frame = [(vehicle_id, frame.get(vehicle_id)) for vehicle_id in frame.keys()]

        # Get the boundaries of each of the vehicles.
        vehicles_in_frame_boxes = [(v_id, util.get_box(vehicle)) for v_id, vehicle in vehicles_in_frame]

        # Use each of the vehicles in the frame as ego vehicles in turn.
        for ego_id, ego_vehicle in vehicles_in_frame:

            # We only want to compute the occlusions for non-parked vehicles.
            if self.episode.agents[ego_id].parked():
                continue

            ego_position = ego_vehicle.position
            other_vehicles_boxes = [vehicle_box for v_id, vehicle_box in vehicles_in_frame_boxes if v_id != ego_id]

            # Sort the obstacles based on their distance to the ego. Closer objects may hide other objects, which we
            # then don't need to consider.
            other_vehicles_boxes.sort(key=lambda box: math.dist(ego_position, box.center))
            obstacles = self.buildings + [list(box.boundary) for box in other_vehicles_boxes]

            # Get for that ego vehicle, what areas of each lane are occluded.
            ego_occluded_lanes = self.get_occlusions_ego_by_road(ego_position, obstacles)

            if self.debug and self.save_format == "p":
                util.plot_map(self.scenario_map, self.scenario_config, frame=frame)
                util.plot_occlusions(ego_position, self.occlusion_lines, ego_occluded_lanes,
                                     non_visible_areas=self.occlusions_far_away)
                self.occlusion_lines = []
                util.show_plot()

            frame_occlusions[ego_id] = ego_occluded_lanes
        return frame_occlusions

    def get_occlusions_ego_by_road(self, ego_position, obstacles):
        """
        Get all the occlusions inside each possible lane in the map.

        Args:
            ego_position:         position of the ego vehicle
            obstacles:            list of boundaries of the obstacles that could create occlusions

        Returns:
             A dictionary with the road id as key and another dictionary as value. The latter dictionary has the
             road's lanes id as key and a Multipolygon as value to represent the occluded areas.

        """
        occlusions_by_roads = {k: {} for k in self.scenario_map.roads.keys()}

        # First find all the occluded areas.
        occluded_areas = self.get_occlusions_ego(ego_position, obstacles)

        # Add to the occlusions everything that is more than OCCLUSION_RADIUS meters away.
        non_visible_areas = self._get_occlusions_far_away(ego_position)
        occluded_areas = unary_union([occluded_areas, non_visible_areas])

        # Store all the occlusions w.r.t the ego.
        occlusions_by_roads["occlusions"] = self._get_format(occluded_areas)

        # Find what areas in each lane is occluded.
        for road in self.scenario_map.roads.values():
            all_road_occlusions = road.boundary.buffer(0).intersection(occluded_areas)
            road_occlusions = {"occlusions": self._get_format(all_road_occlusions)}

            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    if lane.id == 0 or lane.type != "driving":
                        continue

                    intersection = lane.boundary.buffer(0).intersection(occluded_areas)
                    road_occlusions[lane.id] = self._get_format(intersection)
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

        if self.debug and self.save_format == "p":
            self.occlusions_far_away = non_visible_areas
        return non_visible_areas

    def get_occlusions_ego(self, ego_position, obstacles):
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

            if self.debug and self.save_format == "p":
                self.occlusion_lines.append([(v1, v3), (v2, v4)])

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


