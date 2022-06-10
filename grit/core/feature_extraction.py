from typing import List, Dict, Union, Tuple

import numpy as np
import json, math
from igp2 import AgentState, Lane, VelocityTrajectory, StateTrajectory, Map
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, split

from grit.core.goal_generator import TypedGoal, GoalGenerator
import matplotlib.pyplot as plt


class FeatureExtractor:

    MAX_ONCOMING_VEHICLE_DIST = 100

    # Minimum area the occlusion must have to contain a vehicle (todo: assuming 4m*3m)
    MIN_OCCLUSION_AREA = 12

    # Maximum distance the occlusion can be to be considered as significant for creating occlusions.
    MAX_OCCLUSION_DISTANCE = 30

    FRAME_STEP_SIZE = 25
    MISSING = True
    NON_MISSING = False

    feature_names = {'path_to_goal_length': 'scalar',
                     'in_correct_lane': 'binary',
                     'speed': 'scalar',
                     'acceleration': 'scalar',
                     'angle_in_lane': 'scalar',
                     'vehicle_in_front_dist': 'scalar',
                     'vehicle_in_front_speed': 'scalar',
                     'oncoming_vehicle_dist': 'scalar',
                     'oncoming_vehicle_speed': 'scalar',
                     'road_heading': 'scalar',
                     'exit_number': 'scalar'}

    indicator_features = ['exit_number_`missing`']
    possibly_missing_features = {'exit_number': 'exit_number_missing'}

    def __init__(self, scenario_map: Map, *args):
        self.scenario_map = scenario_map

        # If we want to consider occlusions, we need to provide the scenario map and episode index as parameter,
        # in this order.
        if len(args) > 1:
            self.scenario_name = args[0]
            self.episode_idx = args[1]
            with open(f"occlusions/{self.scenario_name}_e{self.episode_idx}.json", 'r') as json_file:
                self.occlusions = json.load(json_file)

    def extract(self, agent_id: int, frames: List[Dict[int, AgentState]], goal: TypedGoal, ego_agent_id: int = None,
                initial_frame: Dict[int, AgentState] = None) \
            -> Dict[str, Union[float, bool]]:
        """Extracts a dict of features describing the observation

        Args:
            agent_id: identifier for the agent of which we want the features
            frames: list of observed frames
            goal:  goal of the agent
            ego_agent_id: id of the ego agent from whose pov the occlusions are taken. Used for indicator features only
            initial_frame: first frame in which the target agent is visible to the ego. Used for indicator features

        Returns: dict of features values

        """

        current_frame = frames[-1]
        current_state = current_frame[agent_id]
        initial_state = frames[0][agent_id]
        current_lane = goal.lane_path[0]
        lane_path = goal.lane_path

        speed = current_state.speed
        acceleration = np.linalg.norm(current_state.acceleration)
        in_correct_lane = self.in_correct_lane(lane_path)
        path_to_goal_length = self.path_to_goal_length(current_state, goal, lane_path)
        angle_in_lane = self.angle_in_lane(current_state, current_lane)
        road_heading = self.road_heading(lane_path)
        exit_number = self.exit_number(initial_state, lane_path)
        goal_type = goal.goal_type

        vehicle_in_front_id, vehicle_in_front_dist = self.vehicle_in_front(agent_id, lane_path, current_frame)
        if vehicle_in_front_id is None:
            vehicle_in_front_speed = 20
            vehicle_in_front_dist = 100
        else:
            vehicle_in_front = current_frame[vehicle_in_front_id]
            vehicle_in_front_speed = vehicle_in_front.speed

        oncoming_vehicle_id, oncoming_vehicle_dist = self.oncoming_vehicle(agent_id, lane_path, current_frame)
        if oncoming_vehicle_id is None:
            oncoming_vehicle_speed = 20
        else:
            oncoming_vehicle_speed = current_frame[oncoming_vehicle_id].speed

        base_features = {'path_to_goal_length': path_to_goal_length,
                         'in_correct_lane': in_correct_lane,
                         'speed': speed,
                         'acceleration': acceleration,
                         'angle_in_lane': angle_in_lane,
                         'vehicle_in_front_dist': vehicle_in_front_dist,
                         'vehicle_in_front_speed': vehicle_in_front_speed,
                         'oncoming_vehicle_dist': oncoming_vehicle_dist,
                         'oncoming_vehicle_speed': oncoming_vehicle_speed,
                         'road_heading': road_heading,
                         'exit_number': exit_number,
                         'goal_type': goal_type}

        # We pass the ego_agent_id only if we want to extract the indicator_features.
        if ego_agent_id:

            frame_id = math.ceil(current_state.time / self.FRAME_STEP_SIZE)
            frame_occlusions = self.occlusions[str(frame_id)]

            occlusions = unary_union(self.get_occlusions_ego(frame_occlusions, ego_agent_id))

            vehicle_in_front_occluded = self.is_vehicle_in_front_missing(ego_agent_id, agent_id, lane_path,
                                                                         current_frame, occlusions)

            oncoming_vehicle_occluded = self.is_oncoming_vehicle_missing(ego_agent_id, lane_path, current_frame,
                                                                         occlusions)

            initial_state = initial_frame[agent_id]

            exit_number_occluded = self.is_exit_number_missing(initial_state, goal)

            indicator_features = {'vehicle_in_front_dist': vehicle_in_front_occluded,
                                  'vehicle_in_front_speed': vehicle_in_front_occluded,
                                  'oncoming_vehicle_dist': oncoming_vehicle_occluded,
                                  'oncoming_vehicle_speed': oncoming_vehicle_occluded,
                                  'exit_number': exit_number_occluded}

            return base_features.update(indicator_features)
        return base_features

    @staticmethod
    def get_vehicles_in_route(ego_agent_id: int, path: List[Lane], frame: Dict[int, AgentState]):
        agents = []
        for agent_id, agent in frame.items():
            agent_point = Point(*agent.position)
            if agent_id != ego_agent_id:
                for lane in path:
                    if lane.boundary.contains(agent_point):
                        agents.append(agent_id)
        return agents

    @staticmethod
    def angle_in_lane(state: AgentState, lane: Lane) -> float:
        """
        Get the signed angle between the vehicle heading and the lane heading
        Args:
            state: current state of the vehicle
            lane: : current lane of the vehicle

        Returns: angle in radians
        """
        lon = lane.distance_at(state.position)
        lane_heading = lane.get_heading_at(lon)
        angle_diff = np.diff(np.unwrap([lane_heading, state.heading]))[0]
        return angle_diff

    @staticmethod
    def road_heading(lane_path: List[Lane]):
        lane = lane_path[-1]
        start_heading = lane.get_heading_at(0)
        end_heading = lane.get_heading_at(lane.length)
        heading_change = np.diff(np.unwrap([start_heading, end_heading]))[0]
        return heading_change

    @staticmethod
    def in_correct_lane(lane_path: List[Lane]):
        for idx in range(0, len(lane_path) - 1):
            if lane_path[idx].lane_section == lane_path[idx+1].lane_section:
                return False
        return True

    @classmethod
    def path_to_goal_length(cls, state: AgentState, goal: TypedGoal, path: List[Lane]) -> float:
        end_point = goal.goal.center
        return cls.path_to_point_length(state, end_point, path)

    @classmethod
    def vehicle_in_front(cls, target_agent_id: int, lane_path: List[Lane], frame: Dict[int, AgentState]):
        state = frame[target_agent_id]
        vehicles_in_route = cls.get_vehicles_in_route(target_agent_id, lane_path, frame)
        min_dist = np.inf
        vehicle_in_front = None
        target_dist_along = cls.dist_along_path(lane_path, state.position)

        # find vehicle in front with closest distance
        for agent_id in vehicles_in_route:
            agent_point = frame[agent_id].position
            agent_dist = cls.dist_along_path(lane_path, agent_point)
            dist = agent_dist - target_dist_along
            if 1e-4 < dist < min_dist:
                vehicle_in_front = agent_id
                min_dist = dist

        return vehicle_in_front, min_dist

    def is_vehicle_in_front_missing(self, target_id: int, lane_path: List[Lane], frame: Dict[int, AgentState],
                                    occlusions: MultiPolygon):
        """
        Args:
            target_id:  id of the vehicle for which we are extracting the features
            lane_path:  lanes executed by the target vehicle if it had the assigned goal
            frame:      current state of the world
            occlusions: must be unary union of all the occlusions for the ego at that point in time
        """
        vehicle_in_front, dist = self.vehicle_in_front(target_id, lane_path, frame)

        target_state = frame[target_id]
        current_lane = self.scenario_map.best_lane_at(target_state.position, target_state.heading, True)

        target_point = Point(*target_state.position)

        midline = current_lane.midline
        crossing_point_on_midline = midline.interpolate(midline.project(target_point)).buffer(0.0001)

        # Remove all the occlusions that are behind the target vehicle as we want possible hidden vehicles in front.
        _, _, area_after = split(midline, crossing_point_on_midline)

        occlusions = self.get_occlusions_past_point(current_lane, lane_path, occlusions, target_point, area_after)

        if occlusions is None:
            return self.NON_MISSING

        occlusions = self.get_significant_occlusions(occlusions)

        if isinstance(occlusions, Polygon):
            plt.plot(*occlusions.exterior.xy, color="b")
        elif isinstance(occlusions, MultiPolygon):
            [plt.plot(*lane.exterior.xy, color="b") for lane in occlusions.geoms]

        plt.show()

        if occlusions:
            distance_to_occlusion = occlusions.distance(target_point)

            if distance_to_occlusion > self.MAX_OCCLUSION_DISTANCE:
                # The occlusion is far away, and won't affect the target vehicle decisions.
                return self.NON_MISSING

            # Otherwise, the feature is not missing if there is a vehicle closer to the target than to the occlusions.
            return not dist < distance_to_occlusion

        else:
            # "occlusions" is not None only if they are large enough to fit a hidden vehicle. Then, the feature is not
            # missing
            return self.NON_MISSING

    @classmethod
    def dist_along_path(cls, path: List[Lane], point: np.ndarray):
        shapely_point = Point(*point)
        midline = cls.get_lane_path_midline(path)
        dist = midline.project(shapely_point)
        return dist

    @staticmethod
    def get_current_path_lane_idx(path: List[Lane], point: np.ndarray) -> int:
        """ Get the index of the lane closest to a point"""
        if type(point) == Point:
            shapely_point = point
        else:
            shapely_point = Point(point[0], point[1])

        for idx, lane in enumerate(path):
            if lane.boundary.contains(shapely_point):
                return idx

        closest_lane_dist = np.inf
        closest_lane_idx = None
        for idx, lane in enumerate(path):
            dist = lane.boundary.exterior.distance(shapely_point)
            if dist < closest_lane_dist:
                closest_lane_dist = dist
                closest_lane_idx = idx
        return closest_lane_idx

    @staticmethod
    def path_to_point_length(state: AgentState, point: np.ndarray, path: List[Lane]) -> float:
        """ Get the length of a path across multiple lanes

        Args:
            state: initial state of the vehicle
            point: final point to be reached
            path: sequence of lanes traversed

        Returns: path length

        """
        end_lane = path[-1]
        end_lane_dist = end_lane.distance_at(point)

        start_point = state.position
        start_lane = path[0]
        start_lane_dist = start_lane.distance_at(start_point)

        dist = end_lane_dist - start_lane_dist
        if len(path) > 1:
            prev_lane = None
            for idx in range(len(path) - 1):
                lane = path[idx]
                lane_change = prev_lane is not None and prev_lane.lane_section == lane.lane_section
                if not lane_change:
                    dist += lane.length
                prev_lane = lane
        return dist

    @staticmethod
    def angle_to_goal(state, goal):
        goal_heading = np.arctan2(goal[1] - state.y, goal[0] - state.x)
        return np.diff(np.unwrap([goal_heading, state.heading]))[0]

    @staticmethod
    def get_junction_lane(lane_path: List[Lane]) -> Union[Lane, None]:
        for lane in lane_path:
            if lane.parent_road.junction is not None:
                return lane
        return None

    @staticmethod
    def get_lane_path_midline(lane_path: List[Lane]) -> LineString:
        midline_points = []
        for idx, lane in enumerate(lane_path[:-1]):
            # check if next lane is adjacent
            if lane_path[idx + 1] not in lane.lane_section.all_lanes:
                midline_points.extend(lane.midline.coords[:-1])
        midline_points.extend(lane_path[-1].midline.coords)
        lane_ls = LineString(midline_points)
        return lane_ls

    # todo: refactor function
    def _get_oncoming_vehicles(self, lane_path: List[Lane], ego_agent_id: int, frame: Dict[int, AgentState],
                               occlusions: MultiPolygon = None, check_occlusions: bool = False) \
            -> Dict[int, Tuple[AgentState, float]]:
        oncoming_vehicles = {}

        ego_junction_lane = self.get_junction_lane(lane_path)
        if ego_junction_lane is None:
            return oncoming_vehicles
        ego_junction_lane_boundary = ego_junction_lane.boundary.buffer(0)
        lanes_to_cross = self._get_lanes_to_cross(ego_junction_lane)

        agent_lanes = [(i, self.scenario_map.best_lane_at(s.position, s.heading, True)) for i, s in frame.items()]

        crossing_points = []
        occluded_oncoming_areas = []
        occlusion_start_dists = []

        for lane_to_cross in lanes_to_cross:
            crossing_point = lane_to_cross.boundary.buffer(0).intersection(ego_junction_lane_boundary).centroid
            crossing_points.append(crossing_point)
            lane_sequence = self._get_predecessor_lane_sequence(lane_to_cross)
            midline = self.get_lane_path_midline(lane_sequence)
            crossing_lon = midline.project(crossing_point)

            # Find the occlusions on the lanes that the ego vehicle will cross.
            if occlusions:

                # Ignore the occlusions that are on the "opposite" (w.r.t traffic direction) side of the crossing point.
                # We only want to check if there is a hidden vehicle that could collide with the ego.
                # This can only happen with vehicles that are driving in the lane's direction of traffic
                # and have not passed the crossing point that the ego will drive through.
                crossing_point_on_midline = midline.interpolate(crossing_lon).buffer(0.0001)

                # Get the part of the midline of the lanes in which there could be oncoming vehicles, that is before
                # the crossing point.
                area_before, _, area_after = split(midline, crossing_point_on_midline)

                # Get the significant occlusions.
                lane_occlusions = self.get_occlusions_past_point(ego_junction_lane,
                                                                 lane_sequence,
                                                                 occlusions,
                                                                 crossing_point,
                                                                 area_before)

                if lane_occlusions is None:
                    continue

                occluded_oncoming_areas.append(lane_occlusions)

        if occluded_oncoming_areas:
            occluded_oncoming_areas = unary_union(occluded_oncoming_areas)

            # Only take the occlusions that could fit a hidden vehicle.
            occluded_oncoming_areas = self.get_significant_occlusions(occluded_oncoming_areas)

        elif check_occlusions:
            # The feature cannot be missing as there are no occlusions.
            # We consider the occlusions to be infinitely away.
            return oncoming_vehicles, np.inf

        for i, lane_to_cross in enumerate(lanes_to_cross):
            lane_sequence = self._get_predecessor_lane_sequence(lane_to_cross)

            midline = self.get_lane_path_midline(lane_sequence)
            plt.plot(*midline.coords.xy, color="g")  # todo

            crossing_point = crossing_points[i]
            crossing_lon = midline.project(crossing_point)

            if occluded_oncoming_areas is not None:
                if isinstance(occluded_oncoming_areas, Polygon):
                    plt.plot(*occluded_oncoming_areas.exterior.xy, color="b")
                elif isinstance(occluded_oncoming_areas, MultiPolygon):
                    [plt.plot(*lane.exterior.xy, color="b") for lane in occluded_oncoming_areas.geoms]
                plt.plot(*crossing_point.coords.xy, marker="x", color="b")

            if check_occlusions:
                occlusion_start_dist = crossing_point.distance(occluded_oncoming_areas)
                occlusion_start_dists.append(occlusion_start_dist)

            # find agents in lane to cross
            for agent_id, agent_lane in agent_lanes:
                agent_state = frame[agent_id]
                if agent_id != ego_agent_id and agent_lane in lane_sequence:
                    agent_lon = midline.project(Point(agent_state.position))
                    dist = crossing_lon - agent_lon

                    if 0 < dist < self.MAX_ONCOMING_VEHICLE_DIST:
                        oncoming_vehicles[agent_id] = (agent_state, dist)

        if check_occlusions:
            return oncoming_vehicles, min(occlusion_start_dists)

        return oncoming_vehicles

    def _get_lanes_to_cross(self, ego_lane: Lane) -> List[Lane]:
        ego_road = ego_lane.parent_road
        ego_incoming_lane = ego_lane.link.predecessor[0]
        ego_lane_boundary = ego_lane.boundary.buffer(0)
        lanes = []
        for connection in ego_road.junction.connections:
            for lane_link in connection.lane_links:
                lane = lane_link.to_lane
                same_predecessor = (ego_incoming_lane.id == lane_link.from_id
                                    and ego_incoming_lane.parent_road.id == connection.incoming_road.id)
                if not (same_predecessor or self._has_priority(ego_road, lane.parent_road)):
                    overlap = ego_lane_boundary.intersection(lane.boundary.buffer(0))
                    if overlap.area > 1:
                        lanes.append(lane)
        return lanes

    def get_occlusions_past_point(self, current_lane, other_lanes, all_occlusions, point_of_cut, area_to_keep):
        """
        Args:
            area_to_keep: part of the MIDLINE we want to keep todo: make clearer
        """

        possible_occlusions = []
        for lane in other_lanes:
            o = all_occlusions.intersection(lane.boundary)
            if isinstance(o, MultiPolygon):
                possible_occlusions.extend(list(o.geoms))
            elif isinstance(o, Polygon):
                possible_occlusions.append(o)
            # We don't consider points or lines are they are not large enough.

        possible_occlusions = unary_union(possible_occlusions)

        if possible_occlusions.area == 0:
            return None

        # Remove all the occlusions that are behind the target vehicle.
        ds = current_lane.boundary.boundary.project(point_of_cut)
        p = current_lane.boundary.boundary.interpolate(ds)

        slope = (p.y - point_of_cut.y) / (p.x - point_of_cut.x)

        import sympy
        s_p = sympy.Point(p.x, p.y)

        direction1 = Point(p.x - point_of_cut.x, p.y - point_of_cut.y)
        direction2 = Point(point_of_cut.x - p.x, point_of_cut.y - p.y)
        p1 = self.get_extended_point(30, slope, direction1, s_p)
        p2 = self.get_extended_point(30, slope, direction2, s_p)

        # Split the occluded areas along the line perpendicular to the current lane and passing through the point_of_cut
        line = LineString([Point(p1.x, p1.y), Point(p2.x, p2.y)])
        intersections = split(possible_occlusions, line)  # todo: rename variables

        return unary_union([intersection for intersection in intersections.geoms
                            if intersection.intersection(area_to_keep).length > 1])

    def get_significant_occlusions(self, occlusions):
        if isinstance(occlusions, MultiPolygon):
            return unary_union([occlusion for occlusion in occlusions.geoms
                                if occlusion.area > self.MIN_OCCLUSION_AREA])
        elif isinstance(occlusions, Polygon):
            return occlusions if occlusions.area > self.MIN_OCCLUSION_AREA else None

    # todo: fix circular import and use the function in occlusion_detection_geometry.py
    @staticmethod
    def get_extended_point(length, slope, direction, point):
        import math

        delta_x = math.sqrt(length ** 2 / (1 + slope ** 2))
        delta_y = math.sqrt(length ** 2 - delta_x ** 2)

        if direction.x < 0:
            delta_x = -delta_x

        if direction.y < 0:
            delta_y = -delta_y

        return point.translate(delta_x, delta_y)

    @staticmethod
    def get_occlusions_ego(frame_occlusions, ego_id):
        for vehicle_occlusions in frame_occlusions:
            if vehicle_occlusions["ego_agent_id"] == ego_id:
                occlusions_vehicle_frame = vehicle_occlusions["occlusions"]

        occlusions = []
        for road_occlusions in occlusions_vehicle_frame:
            for lane_occlusions in occlusions_vehicle_frame[road_occlusions]:
                for lane_occlusion in occlusions_vehicle_frame[road_occlusions][lane_occlusions]:
                    occlusions.append(Polygon(list(zip(*lane_occlusion))))
        return occlusions

    @classmethod
    def _get_predecessor_lane_sequence(cls, lane: Lane) -> List[Lane]:
        lane_sequence = []
        total_length = 0
        while lane is not None and total_length < 100:
            lane_sequence.insert(0, lane)
            total_length += lane.midline.length
            lane = lane.link.predecessor[0] if lane.link.predecessor else None
        return lane_sequence

    @staticmethod
    def _has_priority(ego_road, other_road):
        for priority in ego_road.junction.priorities:
            if (priority.high_id == ego_road.id
                    and priority.low_id == other_road.id):
                return True
        return False

    def oncoming_vehicle(self, ego_agent_id: int, lane_path: List[Lane], frame: Dict[int, AgentState], max_dist=100):
        oncoming_vehicles = self._get_oncoming_vehicles(lane_path, ego_agent_id, frame)
        min_dist = max_dist
        closest_vehicle_id = None
        for agent_id, (agent, dist) in oncoming_vehicles.items():
            if dist < min_dist:
                min_dist = dist
                closest_vehicle_id = agent_id
        return closest_vehicle_id, min_dist

    def is_oncoming_vehicle_missing(self, target_vehicle_id: int, lane_path: List[Lane], frame: Dict[int, AgentState],
                                    occlusions: MultiPolygon):

        oncoming_vehicles, min_occlusion_distance = self._get_oncoming_vehicles(lane_path, target_vehicle_id, frame,
                                                                                occlusions, check_occlusions=True)

        # If the closest occlusion is too far away (or missing), we say that occlusion is not significant.
        if min_occlusion_distance > self.MAX_OCCLUSION_DISTANCE:
            return False

        # Find the vehicle that is closest to the crossing points.
        min_dist = self.MAX_OCCLUSION_DISTANCE
        for agent_id, (agent, dist) in oncoming_vehicles.items():
            if dist < min_dist:
                min_dist = dist

        # If there is a vehicle that is further away to any of the crossing points that the occlusion, then the feature
        # is missing.
        return min_occlusion_distance < min_dist + 2.5  # the 2.5 meters are in case the vehicle is partially occluded

    def exit_number(self, initial_state: AgentState, future_lane_path: List[Lane]):
        # get the exit number in a roundabout
        if (future_lane_path[-1].parent_road.junction is None
                or future_lane_path[-1].parent_road.junction.junction_group is None
                or future_lane_path[-1].parent_road.junction.junction_group.type != 'roundabout'):
            return 0

        position = initial_state.position
        heading = initial_state.heading
        possible_lanes = self.scenario_map.lanes_within_angle(position, heading, np.pi / 4,
                                                         drivable_only=True, max_distance=3)
        initial_lane = possible_lanes[GoalGenerator.get_best_lane(possible_lanes, position, heading)]

        lane_path = self.path_to_lane(initial_lane, future_lane_path[-1])

        # iterate through lane path and count number of junctions
        exit_number = 0
        entrance_passed = False
        if lane_path is not None:
            for lane in lane_path:
                if self.is_roundabout_entrance(lane):
                    entrance_passed = True
                elif entrance_passed and self.is_roundabout_junction(lane):
                    exit_number += 1

        return exit_number

    def is_exit_number_missing(self, initial_state: AgentState, goal):

        return self.exit_number(initial_state, goal.lane_path) == 0

    @staticmethod
    def is_roundabout_junction(lane: Lane):
        junction = lane.parent_road.junction
        return (junction is not None and junction.junction_group is not None
                and junction.junction_group.type == 'roundabout')

    def is_roundabout_entrance(self, lane: Lane) -> bool:
        predecessor_in_roundabout = (lane.link.predecessor is not None and len(lane.link.predecessor) == 1
                                   and self.scenario_map.road_in_roundabout(lane.link.predecessor[0].parent_road))
        return self.is_roundabout_junction(lane) and not predecessor_in_roundabout

    def get_typed_goals(self, trajectory: VelocityTrajectory, goals: List[Tuple[int, int]]):
        typed_goals = []
        goal_gen = GoalGenerator()
        gen_goals = goal_gen.generate(self.scenario_map, trajectory)
        for goal in goals:
            for gen_goal in gen_goals:
                if gen_goal.goal.reached(Point(*goal)):
                    break
            else:
                gen_goal = None
            typed_goals.append(gen_goal)
        return typed_goals

    @staticmethod
    def goal_type(route: List[Lane]):
        return GoalGenerator.get_juction_goal_type(route[-1])

    @staticmethod
    def path_to_lane(initial_lane: Lane, target_lane: Lane, max_depth=20) -> List[Lane]:
        visited_lanes = {initial_lane}
        open_set = [[initial_lane]]

        while len(open_set) > 0:
            lane_sequence = open_set.pop(0)
            if len(lane_sequence) > max_depth:
                break

            lane = lane_sequence[-1]
            if lane == target_lane:
                return lane_sequence

            junction = lane.parent_road.junction

            neighbours = lane.traversable_neighbours()
            for neighbour in neighbours:
                if neighbour not in visited_lanes:
                    visited_lanes.add(neighbour)
                    open_set.append(lane_sequence + [neighbour])

        return None


class GoalDetector:
    """ Detects the goals of agents based on their trajectories"""

    def __init__(self, possible_goals, dist_threshold=1.5):
        self.dist_threshold = dist_threshold
        self.possible_goals = possible_goals

    def detect_goals(self, trajectory: StateTrajectory):
        goals = []
        goal_frame_idxes = []
        for point_idx, agent_point in enumerate(trajectory.path):
            for goal_idx, goal_point in enumerate(self.possible_goals):
                dist = np.linalg.norm(agent_point - goal_point)
                if dist <= self.dist_threshold and goal_idx not in goals:
                    goals.append(goal_idx)
                    goal_frame_idxes.append(point_idx)
        return goals, goal_frame_idxes

    def get_agents_goals_ind(self, tracks, static_info, meta_info, map_meta, agent_class='car'):
        goal_locations = map_meta.goals
        agent_goals = {}
        for track_idx in range(len(static_info)):
            if static_info[track_idx]['class'] == agent_class:
                track = tracks[track_idx]
                agent_goals[track_idx] = []

                for i in range(static_info[track_idx]['numFrames']):
                    point = np.array([track['xCenter'][i], track['yCenter'][i]])
                    for goal_idx, loc in enumerate(goal_locations):
                        dist = np.linalg.norm(point - loc)
                        if dist < self.dist_threshold and loc not in agent_goals[track_idx]:
                            agent_goals[track_idx].append(loc)
        return agent_goals
