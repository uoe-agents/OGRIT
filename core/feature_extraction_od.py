from typing import List, Dict, Union, Tuple

import numpy as np
from igp2.agents.agentstate import AgentState
from igp2.opendrive.elements.road_lanes import Lane
from igp2.opendrive.map import Map
from shapely.geometry import Point, LineString

from core.goal_generator import TypedGoal


class FeatureExtractor:

    MAX_ONCOMING_VEHICLE_DIST = 100

    feature_names = {'path_to_goal_length': 'scalar',
                     'in_correct_lane': 'binary',
                     'speed': 'scalar',
                     'acceleration': 'scalar',
                     'angle_in_lane': 'scalar',
                     'vehicle_in_front_dist': 'scalar',
                     'vehicle_in_front_speed': 'scalar',
                     'oncoming_vehicle_dist': 'scalar',
                     'oncoming_vehicle_speed': 'scalar'}

    def __init__(self, scenario_map: Map):
        self.scenario_map = scenario_map

    def extract(self, agent_id: int, frames: List[Dict[int, AgentState]], goal: TypedGoal) \
            -> Dict[str, Union[float, bool]]:
        """Extracts a dict of features describing the observation

        Args:
            agent_id: identifier for the agent
            frames: list of observed frames
            goal:  goal of the agent

        Returns: dic of features values

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

        return {'path_to_goal_length': path_to_goal_length,
                'in_correct_lane': in_correct_lane,
                'speed': speed,
                'acceleration': acceleration,
                'angle_in_lane': angle_in_lane,
                'vehicle_in_front_dist': vehicle_in_front_dist,
                'vehicle_in_front_speed': vehicle_in_front_speed,
                'oncoming_vehicle_dist': oncoming_vehicle_dist,
                'oncoming_vehicle_speed': oncoming_vehicle_speed,
                'goal_type': goal_type}

    @staticmethod
    def get_vehicles_in_route(ego_agent_id: int, path: List[Lane], frame: Dict[int, AgentState]):
        agents = []
        for agent_id, agent in frame.items():
            if agent_id != ego_agent_id:
                for lane in path:
                    if lane.boundary.contains(agent.position):
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
    def vehicle_in_front(cls, ego_agent_id: int, lane_path: List[Lane], frame: Dict[int, AgentState]):
        state = frame[ego_agent_id]
        vehicles_in_route = cls.get_vehicles_in_route(ego_agent_id, lane_path, frame)
        min_dist = np.inf
        vehicle_in_front = None
        ego_dist_along = cls.dist_along_path(lane_path, state.position)

        # find vehicle in front with closest distance
        for agent_id in vehicles_in_route:
            agent_point = frame[agent_id].position
            agent_dist = cls.dist_along_path(lane_path, agent_point)
            dist = agent_dist - ego_dist_along
            if 1e-4 < dist < min_dist:
                vehicle_in_front = agent_id
                min_dist = dist

        return vehicle_in_front, min_dist

    @classmethod
    def dist_along_path(cls, path: List[Lane], point: np.ndarray):
        # get current lane
        current_lane_idx = cls.get_current_path_lane_idx(path, point)
        completed_lane_dist = sum([l.length for l in path[:current_lane_idx]])
        dist = path[current_lane_idx].length + completed_lane_dist
        return dist

    @staticmethod
    def get_current_path_lane_idx(path: List[Lane], point: np.ndarray) -> int:
        """ Get the index of the lane closest to a point"""
        for idx, lane in enumerate(path):
            if lane.boundary.contains(point):
                return idx

        closest_lane_dist = np.inf
        closest_lane_idx = None
        for idx, lane in enumerate(path):
            dist = lane.boundary.exterior.distance(point)
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
            prev_lane = start_lane
            for idx in range(len(path) - 1):
                lane = path[idx]
                lane_change = prev_lane.lane_section == lane.lane_section
                if not lane_change:
                    dist += lane.length
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
        final_point = lane_path[-1].midline.coords[-1]
        midline_points = [p for ll in lane_path for p in ll.midline.coords[:-1]] + [final_point]
        lane_ls = LineString(midline_points)
        return lane_ls

    def _get_oncoming_vehicles(self, lane_path: List[Lane], ego_agent_id: int, frame: Dict[int, AgentState]) \
            -> Dict[int, Tuple[AgentState, float]]:
        oncoming_vehicles = {}

        ego_junction_lane = self.get_junction_lane(lane_path)
        if ego_junction_lane is None:
            return oncoming_vehicles

        lanes_to_cross = self._get_lanes_to_cross(ego_junction_lane)

        agent_lanes = [(i, self.scenario_map.best_lane_at(s.position, s.heading, True)) for i, s in frame.items()]

        for lane_to_cross in lanes_to_cross:
            lane_sequence = self._get_predecessor_lane_sequence(lane_to_cross)
            midline = self.get_lane_path_midline(lane_sequence)
            crossing_point = lane_to_cross.boundary.intersection(ego_junction_lane.boundary).centroid
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
        lanes = []
        for connection in ego_road.junction.connections:
            for lane_link in connection.lane_links:
                lane = lane_link.to_lane
                same_predecessor = (ego_incoming_lane.id == lane_link.from_id
                                    and ego_incoming_lane.parent_road.id == connection.incoming_road.id)
                if not (same_predecessor or self._has_priority(ego_road, lane.parent_road)):
                    overlap = ego_lane.boundary.intersection(lane.boundary)
                    if overlap.area > 1:
                        lanes.append(lane)
        return lanes

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
