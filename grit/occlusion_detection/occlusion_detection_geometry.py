import numpy as np
import math
import pickle
from typing import List, Dict, Tuple
from itertools import combinations

from grit.occlusion_detection.occlusion_line import OcclusionLine as Line
from grit.core.data_processing import get_episode_frames

import igp2 as ip
from igp2 import AgentState
from igp2.data.scenario import InDScenario, ScenarioConfig
from igp2.opendrive.map import Map

import matplotlib.pyplot as plt

from shapely.geometry import MultiPoint, Polygon, MultiPolygon
from shapely.ops import unary_union

# How many meters away from the vehicle do we want to detect occlusions.
OCCLUSION_LINE_LENGTH = 50

# Set up the colors for the plots.
OCCLUSIONS_COLOR = "g"
OCCLUDED_LANE_COLOR = "r"
OBSTACLES_COLOR = "y"

OCCLUDED_AREA_ALPHA = 0.7
OCCLUSION_SEGMENTS_ALPHA = 0.3


class OcclusionDetector2D:

    def __init__(self, scenario_name: str, episode_idx: int, debug: bool = False, debug_steps: bool = False):
        self.scenario_name = scenario_name
        self.episode_idx = episode_idx
        self.scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{self.scenario_name}.xodr")
        self.scenario_config = ScenarioConfig.load(f"scenarios/configs/{self.scenario_name}.json")
        self.scenario = InDScenario(self.scenario_config)
        self.episode = self.scenario.load_episode(episode_idx)
        self.buildings = self.scenario_config.buildings

        # "Debug" mode takes precedence over "debug_steps"
        self.debug = debug
        self.debug_steps = debug_steps if debug_steps and not self.debug else False
        self.occlusion_lines = []

    def extract_occlusions(self):
        # Take a step every 25 recorded frames (1s)
        # episode_frames contain for each second the list of frames for all vehicles alive that moment
        episode_frames = get_episode_frames(self.episode, exclude_parked_cars=False, exclude_bicycles=True, step=25)

        all_occlusion_data = {}

        for frame_id, frame in enumerate(episode_frames):
            print(f"Starting frame {frame_id}/{len(episode_frames) - 1}")

            all_occlusion_data[frame_id] = self.get_occlusions_frame(frame)

        occlusions_file_name = f"occlusions/{self.scenario_name}_e{self.episode_idx}.p"
        with open(occlusions_file_name, 'wb') as file:
            pickle.dump(all_occlusion_data, file)

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
                self.plot_map(frame=frame, obstacles=obstacles+[ego_vehicle_boundary])
                self.plot_occlusions(ego_position, self.occlusion_lines, ego_occluded_lanes)
                self.occlusion_lines = []
                plt.show()

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

            v3 = l1.get_extended_point(OCCLUSION_LINE_LENGTH - l1.length, v1)
            v4 = l2.get_extended_point(OCCLUSION_LINE_LENGTH - l2.length, v2)

            if self.debug or self.debug_steps:
                self.occlusion_lines.append([(v1, v3), (v2, v4)])

            if self.debug_steps:
                self.plot_map(obstacles=obstacles+[ego_vehicle_boundary])
                self.plot_occlusions(ego_position, self.occlusion_lines)
                self.occlusion_lines = []
                plt.show()

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

    def plot_map(self, frame: Dict[int, AgentState] = None, obstacles: List[List[List[float]]] = None):

        if self.scenario_config is not None:
            ip.plot_map(self.scenario_map, markings=False, midline=False, scenario_config=self.scenario_config,
                        plot_background=True, ignore_roads=True, plot_goals=True)
        else:
            ip.plot_map(self.scenario_map, markings=False, midline=False)

        if frame:
            for aid, state in frame.items():
                plt.plot(*state.position, marker="x")
                plt.text(*state.position, aid)

        if obstacles:
            for obstacle in obstacles:
                x, y = list(zip(*obstacle))
                OcclusionDetector2D.plot_area(x, y, color=OBSTACLES_COLOR, contour=True)

    @staticmethod
    def plot_area(x, y, color="r", alpha=.5, linewidth=None, contour=False):
        """
        Given the x and y coordinates of the points defining the polygon, plot the boundaries and shade the interior.
        """
        x = list(x)
        y = list(y)
        xy = np.transpose([x, y])

        plt.gca().add_patch(plt.Polygon(xy, color=color, alpha=alpha, fill=True, linewidth=linewidth))

        if contour:
            x.append(x[0])
            y.append(y[0])
            plt.plot(x, y, color=color)

    @staticmethod
    def plot_occlusions(ego_position: np.array, occlusion_lines: List[List[Tuple[int, int]]],
                        road_occlusions: Dict[int, Dict[int, List[Polygon]]] = None):

        occluded_areas = []
        for occlusion_line in occlusion_lines:
            x0, y0 = ego_position
            ((x1, y1), (x3, y3)), ((x2, y2), (x4, y4)) = occlusion_line

            # Plot the line segment going from the ego vehicle to the vertices of the obstacle.
            plt.plot([x0, x3], [y0, y3], color=OCCLUSIONS_COLOR, alpha=0.5)
            plt.plot([x0, x4], [y0, y4], color=OCCLUSIONS_COLOR, alpha=0.5)

            occluded_areas.append(Polygon([(x1, y1), (x2, y2), (x4, y4), (x3, y3)]))

        OcclusionDetector2D.plot_area_from_list(occluded_areas, color=OCCLUSIONS_COLOR, alpha=OCCLUDED_AREA_ALPHA)

        if road_occlusions is None:
            return

        lane_occlusions_all = []
        for road_id, occlusions in road_occlusions.items():
            for lane_id, lane_occlusions in occlusions.items():

                if not lane_occlusions:
                    continue

                lane_occlusions_all.append(lane_occlusions)
        OcclusionDetector2D.plot_area_from_list(lane_occlusions_all, color=OCCLUDED_LANE_COLOR, alpha=OCCLUDED_AREA_ALPHA)

    @staticmethod
    def plot_area_from_list(geometries, color="r", alpha=0.5):
        geometries = unary_union(geometries)
        if not geometries.is_empty:
            if isinstance(geometries, Polygon):
                OcclusionDetector2D.plot_area(*geometries.exterior.xy, color=color,
                                              alpha=alpha)
            elif isinstance(geometries, MultiPolygon):
                for geometry in geometries.geoms:
                    OcclusionDetector2D.plot_area(*geometry.exterior.xy, color=color,
                                                  alpha=alpha)

