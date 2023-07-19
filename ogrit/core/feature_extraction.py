import math
import pickle
from functools import lru_cache
from typing import List, Dict, Union, Tuple

import numpy as np
from igp2 import AgentState, Lane, VelocityTrajectory, StateTrajectory, Map, Road
from igp2.data import ScenarioConfig
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, split, snap

from ogrit.core.base import get_occlusions_dir, get_scenarios_dir
from ogrit.core.goal_generator import TypedGoal, GoalGenerator


def heading_diff(a: float, b: float) -> float:
    return np.diff(np.unwrap([a, b]))[0]


class FeatureExtractor:
    MAX_ONCOMING_VEHICLE_DIST = 100

    # Minimum area the occlusion must have to contain a vehicle (assuming a 4m*3m vehicle)
    MIN_OCCLUSION_AREA = 12

    # Maximum distance the occlusion can be to be considered as significant for creating occlusions.
    MAX_OCCLUSION_DISTANCE = 30

    MISSING = True
    NON_MISSING = False

    # Include here the features that you want to use in the decision tree.
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
                     'exit_number': 'integer',
                     # 'speed_change_1s': 'scalar',
                     # 'speed_change_2s': 'scalar',
                     # 'speed_change_3s': 'scalar',
                     'heading_change_1s': 'scalar',
                     # 'heading_change_2s': 'scalar',
                     # 'heading_change_3s': 'scalar',
                     # 'dist_travelled_1s': 'scalar',
                     # 'dist_travelled_2s': 'scalar',
                     # 'dist_travelled_3s': 'scalar'
                     'roundabout_slip_road': 'binary',
                     'roundabout_uturn': 'binary',
                     'angle_to_goal': 'scalar'
                     }

    possibly_missing_features = {'exit_number': 'exit_number_missing',
                                 'oncoming_vehicle_dist': 'oncoming_vehicle_missing',
                                 'oncoming_vehicle_speed': 'oncoming_vehicle_missing',
                                 'vehicle_in_front_dist': 'vehicle_in_front_missing',
                                 'vehicle_in_front_speed': 'vehicle_in_front_missing',
                                 'speed': 'target_1s_occluded',
                                 'acceleration': 'target_1s_occluded',
                                 # 'speed_change_1s': 'target_1s_occluded',
                                 # 'speed_change_2s': 'target_2s_occluded',
                                 # 'speed_change_3s': 'target_3s_occluded',
                                 'heading_change_1s': 'target_1s_occluded',
                                 # 'heading_change_2s': 'target_2s_occluded',
                                 # 'heading_change_3s': 'target_3s_occluded',
                                 # 'dist_travelled_1s': 'target_1s_occluded',
                                 # 'dist_travelled_2s': 'target_2s_occluded',
                                 # 'dist_travelled_3s': 'target_3s_occluded',
                                 'roundabout_slip_road': 'exit_number_missing',
                                 'roundabout_uturn': 'exit_number_missing',
                                 }
    indicator_features = list(set(possibly_missing_features.values()))

    def __init__(self, scenario_map: Map, scenario_name=None, episode_idx=None):
        self.scenario_map = scenario_map

        # If we want to consider occlusions, we need to provide the scenario map and episode index as parameter,
        # in this order.
        if scenario_name is not None and episode_idx is not None:
            self.scenario_name = scenario_name
            self.episode_idx = episode_idx
            with open(get_occlusions_dir() + f"{self.scenario_name}_e{self.episode_idx}.p", 'rb') as file:
                self.occlusions = pickle.load(file)

        if scenario_name is not None:
            self.config = ScenarioConfig.load(f"{get_scenarios_dir()}/configs/{scenario_name}.json")
        else:
            self.config = None

    def extract(self, agent_id: int, frames: List[Dict[int, AgentState]], goal: TypedGoal, ego_agent_id: int = None,
                initial_frame: Dict[int, AgentState] = None, target_occlusion_history: List[bool] = None, fps=25) \
            -> Dict[str, Union[float, bool]]:
        """Extracts a dict of features describing the observation

        Args:
            agent_id: identifier for the agent of which we want the features
            frames: list of observed frames
            goal:  goal of the agent
            ego_agent_id: id of the ego agent from whose pov the occlusions are taken. Used for indicator features
            initial_frame: first frame in which the target agent is visible to the ego. Used for indicator features
            target_occlusion_history: list indicating whether the target vehicle was occluded in previous frames
            fps: f

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
        angle_to_goal = self.angle_to_goal(current_state, goal)

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

        speed_change_1s = self.get_speed_change(agent_id, frames, frames_ago=fps)
        speed_change_2s = self.get_speed_change(agent_id, frames, frames_ago=2 * fps)
        speed_change_3s = self.get_speed_change(agent_id, frames, frames_ago=3 * fps)

        heading_change_1s = self.get_heading_change(agent_id, frames, frames_ago=fps)
        heading_change_2s = self.get_heading_change(agent_id, frames, frames_ago=2 * fps)
        heading_change_3s = self.get_heading_change(agent_id, frames, frames_ago=3 * fps)

        dist_travelled_1s = self.get_dist_travelled(agent_id, frames, frames_ago=fps)
        dist_travelled_2s = self.get_dist_travelled(agent_id, frames, frames_ago=2 * fps)
        dist_travelled_3s = self.get_dist_travelled(agent_id, frames, frames_ago=3 * fps)

        roundabout_uturn = self.is_roundabout_uturn(exit_number)
        roundabout_slip_road = self.slip_road(exit_number, goal)

        features = {'path_to_goal_length': path_to_goal_length,
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
                    'goal_type': goal_type,
                    'speed_change_1s': speed_change_1s,
                    'speed_change_2s': speed_change_2s,
                    'speed_change_3s': speed_change_3s,
                    'heading_change_1s': heading_change_1s,
                    'heading_change_2s': heading_change_2s,
                    'heading_change_3s': heading_change_3s,
                    'dist_travelled_1s': dist_travelled_1s,
                    'dist_travelled_2s': dist_travelled_2s,
                    'dist_travelled_3s': dist_travelled_3s,
                    'roundabout_uturn': roundabout_uturn,
                    'roundabout_slip_road': roundabout_slip_road,
                    'angle_to_goal': angle_to_goal,

                    # Note: x, y, heading below are used for the absolute position LSTM baseline and not by OGRIT
                    'x': current_state.position[0],
                    'y': current_state.position[1],
                    'heading': current_state.heading}

        # We pass the ego_agent_id only if we want to extract the indicator features.
        if ego_agent_id is not None:
            occlusions = self.occlusions[current_state.time][ego_agent_id]["occlusions"]
            vehicle_in_front_occluded = self.is_vehicle_in_front_missing(vehicle_in_front_dist, agent_id, lane_path,
                                                                         current_frame, occlusions)

            oncoming_vehicle_occluded = self.is_oncoming_vehicle_missing(oncoming_vehicle_dist, lane_path, occlusions)

            # Get the first state in which both the ego and target vehicles are alive (even if target is occluded).
            initial_state = initial_frame[agent_id]

            exit_number_occluded = self.is_exit_number_missing(initial_state, goal) \
                if goal_type == "exit-roundabout" else False

            target_1s_occluded = self.target_previously_occluded(frames=frames, frames_ago=fps,
                                                                 target_occlusion_history=target_occlusion_history,
                                                                 fps=fps)
            target_2s_occluded = self.target_previously_occluded(frames=frames, frames_ago=2 * fps,
                                                                 target_occlusion_history=target_occlusion_history,
                                                                 fps=fps)
            target_3s_occluded = self.target_previously_occluded(frames=frames, frames_ago=3 * fps,
                                                                 target_occlusion_history=target_occlusion_history,
                                                                 fps=fps)

            indicator_features = {'vehicle_in_front_missing': vehicle_in_front_occluded,
                                  'oncoming_vehicle_missing': oncoming_vehicle_occluded,
                                  'exit_number_missing': exit_number_occluded,
                                  'target_1s_occluded': target_1s_occluded,
                                  'target_2s_occluded': target_2s_occluded,
                                  'target_3s_occluded': target_3s_occluded, }

            features.update(indicator_features)
        return features

    @staticmethod
    def get_speed_change(agent_id: int, frames: List[Dict[int, AgentState]], frames_ago: int) -> float:
        if frames_ago + 1 > len(frames):
            return 0.
        initial_speed = frames[-(frames_ago + 1)][agent_id].speed
        current_speed = frames[-1][agent_id].speed
        speed_change = current_speed - initial_speed
        return speed_change

    @staticmethod
    def target_previously_occluded(frames: List[Dict[int, AgentState]], frames_ago: int,
                                   target_occlusion_history: List[bool], fps=25) -> bool:
        assert frames_ago % fps == 0
        return (frames_ago + 1 > len(frames)
                or len(target_occlusion_history) < frames_ago // fps + 1
                or target_occlusion_history[-(frames_ago // fps + 1)]
                or target_occlusion_history[-1])

    @staticmethod
    def get_heading_change(agent_id: int, frames: List[Dict[int, AgentState]], frames_ago: int) -> float:
        if frames_ago + 1 > len(frames):
            return 0.
        initial_heading = frames[-(frames_ago + 1)][agent_id].heading
        current_heading = frames[-1][agent_id].heading
        heading_change = heading_diff(initial_heading, current_heading)
        return heading_change

    @staticmethod
    def get_dist_travelled(agent_id: int, frames: List[Dict[int, AgentState]], frames_ago: int) -> float:
        if frames_ago + 1 > len(frames):
            return 0.
        initial_pos = frames[-(frames_ago + 1)][agent_id].position
        current_pos = frames[-1][agent_id].position
        dist_travelled = np.linalg.norm(current_pos - initial_pos)
        return dist_travelled

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
        angle_diff = heading_diff(lane_heading, state.heading)
        return angle_diff

    @staticmethod
    def road_heading(lane_path: List[Lane]):
        lane = lane_path[-1]
        start_heading = lane.get_heading_at(0)
        end_heading = lane.get_heading_at(lane.length)
        heading_change = heading_diff(start_heading, end_heading)
        return heading_change

    @staticmethod
    def in_correct_lane(lane_path: List[Lane]):
        for idx in range(0, len(lane_path) - 1):
            if lane_path[idx].lane_section == lane_path[idx + 1].lane_section:
                return False
        return True

    @staticmethod
    def multi_lane(lane: Lane):
        # check if can switch lane - see igp2
        pass

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

        # find the vehicle in front with closest distance
        for agent_id in vehicles_in_route:
            agent_point = frame[agent_id].position
            agent_dist = cls.dist_along_path(lane_path, agent_point)
            dist = agent_dist - target_dist_along
            if 1e-4 < dist < min_dist:
                vehicle_in_front = agent_id
                min_dist = dist

        return vehicle_in_front, min_dist

    def is_vehicle_in_front_missing(self, dist: float, target_id: int, lane_path: List[Lane],
                                    frame: Dict[int, AgentState], occlusions: MultiPolygon):
        """
        Args:
            dist:       distance of the closest oncoming vehicle, if any.
            target_id:  id of the vehicle for which we are extracting the features
            lane_path:  lanes executed by the target vehicle if it had the assigned goal
            frame:      current state of the world
            occlusions: must be unary union of all the occlusions for the ego at that point in time
        """
        target_state = frame[target_id]
        target_point = Point(*target_state.position)
        midline = self.get_lane_path_midline(lane_path)

        # Remove all the occlusions that are behind the target vehicle as we want possible hidden vehicles in front.
        area_before, area_after = self._get_split_at(midline, target_point)
        occlusions = self._get_occlusions_on(occlusions, lane_path, area_after)

        if occlusions is None:
            return self.NON_MISSING

        occlusions = self._get_significant_occlusions(occlusions)

        if occlusions is None:
            # The occlusions are not large enough to hide a vehicle.
            return self.NON_MISSING

        distance_to_occlusion = occlusions.distance(target_point)

        if distance_to_occlusion > self.MAX_OCCLUSION_DISTANCE:
            # The occlusion is far away, and won't affect the target vehicle decisions.
            return self.NON_MISSING

        # Otherwise, the feature is missing if there is an occlusion closer than the vehicle in front.
        return not dist < distance_to_occlusion + 2.5

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
        goal_heading = np.arctan2(goal.goal.center.y - state.position[1], goal.goal.center.x - state.position[0])
        return heading_diff(goal_heading, state.heading)

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

    def _get_split_at(self, midline, point):
        """
        Split the midline at a specific point. The midline should be shorter than two times the MAX_OCCLUSION_DISTANCE
        """

        if midline.length > 2 * self.MAX_OCCLUSION_DISTANCE:
            # if a closed path is unavoidable, using split only does not work
            interpolation_point = midline.interpolate(2 * self.MAX_OCCLUSION_DISTANCE)
            midline = split(snap(midline, interpolation_point, 0.0000000000001), interpolation_point)[0]

        point_on_midline = midline.interpolate(midline.project(point)).buffer(0.0000000000001)

        split_lanes = split(midline, point_on_midline)

        if len(split_lanes) == 2:
            # Handle the case in which the split point is at the start/end of the lane.
            line_before, line_after = split_lanes
        elif len(split_lanes) == 3:
            # The middle segment is due to rounding
            line_before, _, line_after = split_lanes
        return line_before, line_after

    def _get_oncoming_vehicles(self, lane_path: List[Lane], ego_agent_id: int, frame: Dict[int, AgentState]) \
            -> Dict[int, Tuple[AgentState, float]]:
        oncoming_vehicles = {}

        ego_junction_lane = self.get_junction_lane(lane_path)
        if ego_junction_lane is None:
            return oncoming_vehicles
        ego_junction_lane_boundary = ego_junction_lane.boundary.buffer(0)
        lanes_to_cross = self._get_lanes_to_cross(ego_junction_lane)
        agent_lanes = [(i, self.scenario_map.best_lane_at(s.position, s.heading, True)) for i, s in frame.items()]

        for lane_to_cross in lanes_to_cross:
            lane_sequence = self._get_predecessor_lane_sequence(lane_to_cross)
            midline = self.get_lane_path_midline(lane_sequence)
            crossing_point = lane_to_cross.boundary.buffer(0).intersection(ego_junction_lane_boundary).centroid
            crossing_lon = midline.project(crossing_point)

            # find agents in lane to cross
            for agent_id, agent_lane in agent_lanes:
                agent_state = frame[agent_id]
                if agent_id != ego_agent_id and agent_lane in lane_sequence:
                    agent_lon = midline.project(Point(agent_state.position))
                    dist = crossing_lon - agent_lon
                    if 0 < dist < self.MAX_ONCOMING_VEHICLE_DIST:
                        oncoming_vehicles[agent_id] = (agent_state, dist)
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

    @staticmethod
    def _get_occlusions_on(all_occlusions, other_lanes, area_to_keep):
        """
        Get the occlusions that are on the area_to_keep.

        Args:
            all_occlusions: all the occlusions in the current frame
            other_lanes:    lanes for which we want to find the occluded areas
            area_to_keep:   part of the MIDLINE we want the occlusions on
        """

        # Find the occlusions that intersect the lanes we want.
        possible_occlusions = []
        for lane in other_lanes:
            o = all_occlusions.intersection(lane.boundary.buffer(0))

            if isinstance(o, MultiPolygon):
                possible_occlusions.extend(list(o.geoms))
            elif isinstance(o, Polygon):
                possible_occlusions.append(o)

        possible_occlusions = unary_union(possible_occlusions)
        occlusions = Polygon()

        if isinstance(possible_occlusions, Polygon):
            occlusions = possible_occlusions if possible_occlusions.intersection(area_to_keep).length > 1 else Polygon()
        elif isinstance(possible_occlusions, MultiPolygon):
            occlusions = unary_union([occlusion for occlusion in possible_occlusions.geoms
                                      if occlusion.intersection(area_to_keep).length > 1])

        if occlusions.is_empty:
            return None
        return occlusions

    def _get_significant_occlusions(self, occlusions):
        """
        Return a Multipolygon or Polygon with the occlusions that are large enough to fit a hidden vehicle.
        """
        if isinstance(occlusions, MultiPolygon):
            return unary_union([occlusion for occlusion in occlusions.geoms
                                if occlusion.area > self.MIN_OCCLUSION_AREA])
        elif isinstance(occlusions, Polygon):
            return occlusions if occlusions.area > self.MIN_OCCLUSION_AREA else None

    def _get_min_dist_from_occlusions_oncoming_lanes(self, lanes_to_cross,
                                                     ego_junction_lane_boundary, occlusions):
        """
        Get the minimum distance from any of the crossing points to the occlusions that could hide an oncoming vehicle.
        A crossing point is a point along the target vehicle's path inside a junction.

        Args:
            lanes_to_cross:             list of lanes that the target vehicle will intersect while inside the junction.
            ego_junction_lane_boundary: boundary of the ego_junction lane
            occlusions:                 list of all the occlusions in the frame

        """

        occluded_oncoming_areas = []
        crossing_points = []
        for lane_to_cross in lanes_to_cross:
            crossing_point = lane_to_cross.boundary.buffer(0).intersection(ego_junction_lane_boundary).centroid
            crossing_points.append(crossing_point)
            lane_sequence = self._get_predecessor_lane_sequence(lane_to_cross)
            midline = self.get_lane_path_midline(lane_sequence)

            # Find the occlusions on the lanes that the ego vehicle will cross.
            if occlusions:

                # Get the part of the midline of the lanes in which there could be oncoming vehicles, that is before
                # the crossing point.
                # Ignore the occlusions that are "after" (w.r.t traffic direction) the crossing point.
                # We only want to check if there is a hidden vehicle that could collide with the ego.
                # This can only happen with vehicles that are driving in the lane's direction of traffic
                # and have not passed the crossing point that the ego will drive through.
                area_before, area_after = self._get_split_at(midline, crossing_point)
                lane_occlusions = self._get_occlusions_on(occlusions, lane_sequence, area_before)

                if lane_occlusions is not None:
                    occluded_oncoming_areas.append(lane_occlusions)

        if occluded_oncoming_areas:
            occluded_oncoming_areas = unary_union(occluded_oncoming_areas)

            # Only take the occlusions that could fit a hidden vehicle.
            occluded_oncoming_areas = self._get_significant_occlusions(occluded_oncoming_areas)

            # Get the minimum distance from any of the crossing points and the relevant occlusions.
            if occluded_oncoming_areas:
                return min([crossing_point.distance(occluded_oncoming_areas) for crossing_point in crossing_points])

        # If there are no occlusions large enough to fit a hidden vehicle.
        return math.inf

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
    def _has_priority(a: Road, b: Road):
        for priority in a.junction.priorities:
            if (priority.high_id == a.id
                    and priority.low_id == b.id):
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

    def is_oncoming_vehicle_missing(self, min_dist: int, lane_path: List[Lane], occlusions: MultiPolygon):

        ego_junction_lane = self.get_junction_lane(lane_path)
        if ego_junction_lane is None:
            return False
        ego_junction_lane_boundary = ego_junction_lane.boundary.buffer(0)
        lanes_to_cross = self._get_lanes_to_cross(ego_junction_lane)

        min_occlusion_distance = self._get_min_dist_from_occlusions_oncoming_lanes(lanes_to_cross,
                                                                                   ego_junction_lane_boundary,
                                                                                   occlusions)

        # If the closest occlusion is too far away (or missing), we say that occlusion is not significant.
        if min_occlusion_distance > self.MAX_OCCLUSION_DISTANCE:
            return False

        # If the closest oncoming vehicle is further away to any of the crossing points that the occlusion,
        # then the feature is missing. The 2.5 meters offset is in case the vehicle is partially occluded.
        return min_occlusion_distance + 2.5 < min_dist

    def exit_number_round(self, initial_state: AgentState, future_lane_path: List[Lane]):
        # get the exit number in a roundabout from the rounD dataset
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

    def exit_number(self, initial_state: AgentState, future_lane_path: List[Lane]):
        goal_lane = future_lane_path[-1]
        goal_point = goal_lane.midline.coords[-1]
        if (goal_lane.parent_road.junction is None
                or goal_lane.parent_road.junction.junction_group is None
                or goal_lane.parent_road.junction.junction_group.type != 'roundabout'):
            return 0

        position = initial_state.position
        heading = initial_state.heading
        possible_lanes = self.scenario_map.lanes_within_angle(position, heading, np.pi / 4,
                                                              drivable_only=True, max_distance=3)
        initial_lane = possible_lanes[GoalGenerator.get_best_lane(possible_lanes, position, heading)]

        lane_path = self.path_to_lane(initial_lane, future_lane_path[-1])

        # iterate through lane path and find whether the roundabout entrance has been passed
        entrance_passed = False
        if lane_path is not None:
            for lane in lane_path:
                if self.is_roundabout_entrance(lane):
                    entrance_passed = True
                    break
        if not entrance_passed:
            return 0

        goal_generator = GoalGenerator()
        goals = goal_generator.generate_from_state(self.scenario_map, initial_state.position, initial_state.heading)
        goals.sort(key=lambda x: len(x.lane_path))

        exit_number = 0
        for goal_idx, goal in enumerate(goals):
            if goal.goal_type == 'exit-roundabout' and goal.goal.reached(goal_point):
                exit_number = goal_idx + 1
                break
        return exit_number

    def is_exit_number_missing(self, initial_state: AgentState, goal: TypedGoal):
        """
        The exit number feature is missing if we cannot get the exit number. This happens when:
        - the target vehicle is already in the roundabout when it becomes visible to the ego.
        - the target vehicle is occluded w.r.t the ego when it enters the roundabout.

        Args:
            initial_state: state of the target vehicle when it first became visible to the ego
            goal:          the goal we are trying to get the probability for
        """
        return self.exit_number(initial_state, goal.lane_path) == 0

    @staticmethod
    def is_roundabout_junction(lane: Lane):
        junction = lane.parent_road.junction
        return (junction is not None and junction.junction_group is not None
                and junction.junction_group.type == 'roundabout')

    def is_roundabout_entrance(self, lane: Lane) -> bool:
        predecessor_in_roundabout = (lane.link.predecessor is not None and len(lane.link.predecessor) == 1
                                     and self.lane_in_roundabout(lane.link.predecessor[0]))
        return self.is_roundabout_junction(lane) and not predecessor_in_roundabout

    def lane_in_roundabout(self, lane: Lane):
        if self.scenario_map.road_in_roundabout(lane.parent_road):
            return True
        # for openDD maps
        predecessor_roundabout = (lane.link.predecessor is not None
                                  and len(lane.link.predecessor) == 1
                                  and self.is_roundabout_junction(lane.link.predecessor[0]))

        succecessor_roundabout = (lane.link.successor is not None
                                  and len(lane.link.successor) == 1
                                  and self.is_roundabout_junction(lane.link.successor[0]))
        return predecessor_roundabout and succecessor_roundabout

    def get_typed_goals(self, trajectory: VelocityTrajectory, goals: List[Tuple[int, int]], goal_radius=3.5):
        typed_goals = []
        goal_gen = GoalGenerator()
        gen_goals = goal_gen.generate(self.scenario_map, trajectory, goal_radius=goal_radius)
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
    @lru_cache(maxsize=128)
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

    def is_roundabout_uturn(self, exit_number: int) -> bool:
        return exit_number != 0 and self.config is not None and exit_number == len(self.config.goals)

    def slip_road(self, exit_number: int, goal: TypedGoal) -> bool:
        if exit_number == 1 and self.config is not None and self.config.slip_roads is not None:
            for goal_idx, goal_loc in enumerate(self.config.goals):
                if goal.goal.reached(np.array(goal_loc)):
                    return self.config.slip_roads[goal_idx]
        return False


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
