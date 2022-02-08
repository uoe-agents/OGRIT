from collections import defaultdict
from dataclasses import dataclass
from typing import List
import numpy as np
from igp2.opendrive.elements.geometry import normalise_angle
from igp2.trajectory import VelocityTrajectory
from shapely.geometry import Point

from igp2.goal import Goal, PointGoal
from igp2.opendrive.elements.road_lanes import Lane
from igp2.opendrive.map import Map
from igp2.agents.agentstate import AgentState
from igp2.util import Circle


@dataclass
class TypedGoal:
    goal_type: str
    goal: Goal
    lane_path: List[Lane]

    # def __repr__(self):
    #     return f"TypedGoal(goal_type={self.goal_type}, goal={self.goal})"


class GoalGenerator:

    @classmethod
    def generate_goals_from_lane(cls, lane: Lane, scenario_map: Map, visible_region: Circle = None) -> List[TypedGoal]:
        typed_goals = []
        visited_goal_locations = {}
        visited_lanes = {lane}
        open_set = [[lane]]

        while len(open_set) > 0:
            lane_sequence = open_set.pop(0)
            lane = lane_sequence[-1]
            junction = lane.parent_road.junction

            goal_location = lane.midline.coords[-1]
            goal_type = None
            if goal_location not in visited_goal_locations:
                if junction is not None:
                    if junction.junction_group is not None and junction.junction_group.type == 'roundabout':
                        # check if it is a roundabout exit
                        successor_in_roundabout = (len(lane.link.successor) == 1
                            and scenario_map.road_in_roundabout(lane.link.successor[0].parent_road))
                        if not successor_in_roundabout:
                            goal_type = 'exit-roundabout'
                    else:
                        goal_type = cls.get_juction_goal_type(lane)
                elif lane.link.successor is None:
                    goal_type = 'straight-on'
                    # TODO adjacent lanes should share same goal
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
                goal_radius = lane.get_width_at(lane.length) / 2
                goal = PointGoal(goal_location, goal_radius)
                typed_goal = TypedGoal(goal_type, goal, lane_sequence)
                typed_goals.append(typed_goal)

        return typed_goals

    @staticmethod
    def is_roundabout_exit(lane: Lane, scenario_map: Map) -> bool:
        in_roundabout = scenario_map.road_in_roundabout(lane.parent_road)
        successor_in_roundabout = (len(lane.link.successor) == 1
                                   and scenario_map.road_in_roundabout(lane.link.successor[0].parent_road))
        return in_roundabout and not successor_in_roundabout

    def generate(self, scenario_map: Map, trajectory: VelocityTrajectory, visible_region: Circle = None
                 ) -> List[TypedGoal]:
        """ Generate the set of possible goals for an agent state

        Args:
            scenario_map: local road map
            trajectory: trajectory of the vehicle up to the current time
            visible_region: region of the region of the map that is visible

        Returns:
            List of generated goals
        """
        # get list of possible lanes
        # build tree of reachable lanes
        # if junction exit, view radius, or lane end is reached, create goal
        position = trajectory.path[-1]
        heading = trajectory.heading[-1]
        possible_lanes = scenario_map.lanes_within_angle(position, heading, np.pi/4,
                                                         drivable_only=True, max_distance=3)

        lane_goals = [self.generate_goals_from_lane(l, scenario_map, visible_region) for l in possible_lanes]

        # get list of typed goals for each goal locations
        goal_loc_goals = {}
        for goals in lane_goals:
            for goal in goals:
                for goal_loc in goal_loc_goals:
                    if np.allclose(goal_loc, goal.goal.center, atol=1.):
                        goal_loc_goals[goal_loc].append(goal)
                        break
                else:
                    goal_loc_goals[goal.goal.center] = [goal]

        # select best typed goal for each goal location
        typed_goals = []
        for goal_loc, goals in goal_loc_goals.items():
            lanes = [g.lane_path[0] for g in goals]
            best_lane_idx = self.get_best_lane(lanes, position, heading)
            typed_goals.append(goals[best_lane_idx])

        return typed_goals

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
    def get_juction_goal_type(lane: Lane):
        start_heading = lane.get_heading_at(0)
        end_heading = lane.get_heading_at(lane.length)
        heading_change = np.diff(np.unwrap([start_heading, end_heading]))[0]

        if np.pi * 7 / 8 > heading_change > np.pi / 8:
            goal_type = 'turn-left'
        elif -np.pi * 7 / 8 < heading_change < -np.pi / 8:
            goal_type = 'turn-right'
        elif -np.pi / 8 <= heading_change <= np.pi / 8:
            goal_type = 'straight-on'
        else:
            goal_type = 'u-turn'

        return goal_type
