from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import List
import numpy as np
from igp2.opendrive.elements.geometry import normalise_angle
from shapely.geometry import Point
from igp2 import Circle, Goal, PointGoal, Lane, Map, VelocityTrajectory


@dataclass
class TypedGoal:
    goal_type: str
    goal: Goal
    lane_path: List[Lane]

    # def __repr__(self):
    #     return f"TypedGoal(goal_type={self.goal_type}, goal={self.goal})"


class GoalGenerator:

    @classmethod
    def generate_goals_from_lane(cls, lane: Lane, scenario_map: Map, visible_region: Circle = None, goal_radius=3.5) -> List[TypedGoal]:
        typed_goals = []
        visited_lanes = {lane}
        open_set = [[lane]]

        while len(open_set) > 0:
            lane_sequence = open_set.pop(0)
            lane = lane_sequence[-1]
            goal_location = lane.midline.coords[-1]
            goal_type = None
            if cls.is_branching_junction(lane):
                if cls.is_lane_roundabout_junction(lane):
                    if cls.is_roundabout_exit(lane):
                        goal_type = 'exit-roundabout'
                else:
                    goal_type = cls.get_juction_goal_type(lane)
            elif lane.link.successor is None:
                goal_type = 'straight-on'
            elif (visible_region is not None
                  and not visible_region.contains(np.reshape(goal_location, (2, 1))).all()):
                goal_type = 'straight-on'

            if goal_type is None:
                neighbours = lane.traversable_neighbours()
                for neighbour in neighbours:
                    if neighbour not in visited_lanes:
                        visited_lanes.add(neighbour)
                        open_set.append(lane_sequence + [neighbour])
            else:
                goal = PointGoal(goal_location, goal_radius)
                typed_goal = TypedGoal(goal_type, goal, lane_sequence)
                typed_goals.append(typed_goal)

        return typed_goals

    @classmethod
    def is_roundabout_exit(cls, lane: Lane) -> bool:
        if cls.is_lane_roundabout_junction(lane):
            for successor in lane.link.successor:
                if not cls.lane_in_roundabout(successor):
                    cls.lane_in_roundabout(successor)
                    return True
        return False

    @staticmethod
    def is_lane_roundabout_junction(lane: Lane):
        return (lane.parent_road.junction is not None
                and lane.parent_road.junction.junction_group is not None
                and lane.parent_road.junction.junction_group.type == 'roundabout')

    @classmethod
    @lru_cache(maxsize=128)
    def lane_in_roundabout(cls, lane: Lane, search_depth=5) -> bool:
        # checks if lane is in a roundabout, including exits/entrances
        if cls.is_lane_roundabout_junction(lane):
            return True

        successor_id = cls.successor_roundabout_junction_group_id(lane, search_depth)
        predecessor_id = cls.predecessor_roundabout_junction_group_id(lane, search_depth)
        return successor_id is not None and predecessor_id is not None and successor_id == predecessor_id

    @classmethod
    def successor_roundabout_junction_group_id(cls, lane: Lane, search_depth=5):
        if search_depth > 0 and lane.link.successor is not None and len(lane.link.successor) > 0:
            for successor in lane.link.successor:
                if cls.is_lane_roundabout_junction(successor):
                    return successor.parent_road.junction.junction_group.id
                else:
                    return cls.successor_roundabout_junction_group_id(successor, search_depth-1)
        return None

    @classmethod
    def predecessor_roundabout_junction_group_id(cls, lane: Lane, search_depth=5):
        if search_depth > 0 and lane.link.predecessor is not None and len(lane.link.predecessor) > 0:
            for predecessor in lane.link.predecessor:
                if cls.is_lane_roundabout_junction(predecessor):
                    return predecessor.parent_road.junction.junction_group.id
                else:
                    return cls.predecessor_roundabout_junction_group_id(predecessor, search_depth-1)
        return None

    @staticmethod
    def is_branching_junction(lane: Lane):
        if lane.parent_road.junction is None:
            return False
        if len(lane.parent_road.junction.roads) > 2:
            return True

        # search for branching successor
        successor = lane
        while True:
            if successor.link.predecessor is not None and len(successor.link.predecessor) > 1:
                return True
            if successor.parent_road != lane.parent_road:
                break
            if (successor.link.successor is not None
                    and len(successor.link.successor) > 0):
                successor = successor.link.successor[0]
            else:
                break

        # search for branching predecessor
        predecessor = lane
        while True:
            if predecessor.link.successor is not None and len(predecessor.link.successor) > 1:
                return True
            if predecessor.parent_road != lane.parent_road:
                break
            if (predecessor.link.predecessor is not None
                    and len(predecessor.link.predecessor) > 0):
                predecessor = predecessor.link.predecessor[0]
            else:
                break

        return False

    def generate_from_state(self, scenario_map, position, heading, visible_region: Circle = None,
                            goal_radius=3.5) -> List[TypedGoal]:
        possible_lanes = scenario_map.lanes_within_angle(position, heading, np.pi/4,
                                                         drivable_only=True, max_distance=3)

        lane_goals = [self.generate_goals_from_lane(l, scenario_map, visible_region, goal_radius) for l in possible_lanes]

        # get list of typed goals for each goal locations
        goal_loc_goals = {}
        for goals in lane_goals:
            for goal in goals:
                for goal_loc in goal_loc_goals:
                    if np.allclose(goal_loc, goal.goal.center.coords[0], atol=3.5):
                        goal_loc_goals[goal_loc].append(goal)
                        break
                else:
                    goal_loc_goals[goal.goal.center.coords[0]] = [goal]

        # select best typed goal for each goal location
        typed_goals = []
        for goal_loc, goals in goal_loc_goals.items():
            lane_seqs = [g.lane_path for g in goals]
            lanes = self.is_cycle_exist(lane_seqs)
            best_lane_idx = self.get_best_lane(lanes, position, heading)
            typed_goals.append(goals[best_lane_idx])

        return typed_goals

    @staticmethod
    def is_cycle_exist(lane_seqs: List[List[Lane]]) -> List[Lane]:
        """ delete the path if the path is closed. if path is empty, use original path
        Args:
            lane_seqs: list of possible paths to reach goals
        Returns:
            List of possible current lanes
        """
        # check if a cycle exists based on the lane sequence
        lanes = []
        for idx, lane_seq in enumerate(lane_seqs):
            roads = [lane.parent_road for lane in lane_seq]
            cycle_exist = False

            new_roads = [r for r in reversed(roads)]
            for road in roads:
                new_roads.remove(road)
                for ins_road in new_roads:
                    if road.link.predecessor is not None and road.link.predecessor.element.id == ins_road.id:
                        cycle_exist = True
                        break
                if cycle_exist:
                    break
            if not cycle_exist:
                lanes.append(lane_seq[0])
        # in case of a vehicle has to truly drive the entire cycle
        if len(lanes) == 0:
            lanes = [lane_s[0] for lane_s in lane_seqs]
        return lanes

    def generate(self, scenario_map: Map, trajectory: VelocityTrajectory, visible_region: Circle = None,
                 goal_radius=3.5) -> List[TypedGoal]:
        """ Generate the set of possible goals for an agent state

        Args:
            scenario_map: local road map
            trajectory: trajectory of the vehicle up to the current time
            visible_region: region of the region of the map that is visible
            goal_radius: radius of each goal region

        Returns:
            List of generated goals
        """
        # get list of possible lanes
        # build tree of reachable lanes
        # if junction exit, view radius, or lane end is reached, create goal
        position = trajectory.path[-1]
        heading = trajectory.heading[-1]
        return self.generate_from_state(scenario_map, position, heading, visible_region, goal_radius)

    @staticmethod
    def get_best_lane(lanes: List[Lane], position: np.ndarray, heading: float) -> int:
        """ Select the most likely current lane from a list of candidates

        Args:
            lanes: list of candidate lanes
            position: current position of the vehicle
            heading: current heading of the vehicle

        Returns:
            index of the best lane in the list of lanes
        """
        point = Point(position)
        dists = [l.boundary.distance(point) for l in lanes]

        angle_diffs = []
        for lane in lanes:
            road = lane.parent_road
            _, original_angle = road.plan_view.calc(road.midline.project(point))
            # road_start_point = np.array([road.midline.xy[0][0], road.midline.xy[1][0]])
            # road_end_point = np.array([road.midline.xy[0][-1], road.midline.xy[1][-1]])
            # _, road_start_angle = road.plan_view.calc(road.midline.project(Point(road_start_point)))
            # _, road_end_angle = road.plan_view.calc(road.midline.project(Point(road_end_point)))
            # # considering the exit angle for more robust due to road geometry
            # ave_angle = (road_start_angle + road_end_angle) / 2
            if lane.id > 0:
                angle = normalise_angle(original_angle + np.pi)
            else:
                angle = original_angle
            angle_diff = np.abs(normalise_angle(heading - angle))
            angle_diffs.append(angle_diff)

        best_idx = None
        for idx in range(len(lanes)):
            if (best_idx is None
                    or dists[idx] < dists[best_idx]
                    or (dists[idx] == dists[best_idx] and angle_diffs[idx] < angle_diffs[best_idx])):
                best_idx = idx

        return best_idx

    @staticmethod
    def has_priority(a, b):
        for priority in a.junction.priorities:
            if (priority.high_id == a.id
                    and priority.low_id == b.id):
                return True
        return False

    @classmethod
    def is_junction_entry(cls, lane: Lane, direction: str):
        # for priority in lane.parent_road.junction.priorities:
        #     if priority.high_id == lane.parent_road.id:
        #         return False
        # return True

        junction = lane.parent_road.junction
        if lane.link.successor is None:
            return False
        target_successor = lane.link.successor[0]

        for connection in junction.connections:
            for lane_link in connection.lane_links:
                other_lane = lane_link.to_lane
                other_direction = cls.get_lane_direction(other_lane)
                if other_lane.link.successor is not None:
                    other_successor = other_lane.link.successor[0]
                    if (target_successor == other_successor
                            and (other_direction == 'straight' or direction == 'straight')
                            and cls.has_priority(other_lane.parent_road,
                                                 lane.parent_road)):
                        return True
        return False

    @staticmethod
    def get_lane_direction(lane: Lane):
        start_heading = lane.get_heading_at(0)
        end_heading = lane.get_heading_at(lane.length)
        heading_change = np.diff(np.unwrap([start_heading, end_heading]))[0]
        if np.pi * 7 / 8 > heading_change > np.pi / 8:
            return 'left'
        elif -np.pi * 7 / 8 < heading_change < -np.pi / 8:
            return 'right'
        elif -np.pi / 8 <= heading_change <= np.pi / 8:
            return 'straight'
        else:
            return 'backward'

    @classmethod
    def get_juction_goal_type(cls, lane: Lane):
        junction = lane.parent_road.junction

        if junction.junction_group is not None and junction.junction_group.type == 'roundabout':
            return 'exit-roundabout'

        direction = cls.get_lane_direction(lane)
        is_entry = cls.is_junction_entry(lane, direction)
        if direction == 'left':
            return 'enter-left' if is_entry else 'exit-left'
        if direction == 'right':
            return 'enter-right' if is_entry else 'exit-right'
        if direction == 'straight':
            return 'cross-road' if is_entry else 'straight-on'
        return 'u-turn'
