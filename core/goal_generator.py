from dataclasses import dataclass
from typing import List
import numpy as np

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
    def __init__(self, view_radius: float = None):
        """ Initialise a GoalGenerator object

        Args:
            view_radius: radius of the region of the map that is visible
        """
        self.view_radius = view_radius

    def generate(self, scenario_map: Map, state: AgentState) -> List[TypedGoal]:
        """ Generate the set of possible goals for an agent state

        Args:
            scenario_map: local road map
            state: current state of the vehicle

        Returns:
            List of generated goals
        """
        # get list of possible lanes
        # build tree of reachable lanes
        # if junction exit, view radius, or lane end is reached, create goal
        typed_goals = []
        visited_goal_locations = {}
        visited_lanes = set(scenario_map.lanes_at(state.position, drivable_only=True))
        open_set = [[l] for l in visited_lanes]

        while len(open_set) > 0:
            lane_sequence = open_set.pop(0)
            lane = lane_sequence[-1]
            junction = lane.parent_road.junction

            goal_location = lane.midline.coords[-1]
            goal_type = None
            if goal_location not in visited_goal_locations:
                if junction is not None:
                    # TODO check if it's a roundabout
                    goal_type = self.get_juction_goal_type(lane)
                elif lane.link.successor is None:
                    goal_type = 'straight-on'
                    # TODO check if end of lane is outside view radius
                    # TODO adjacent lanes should share same goal

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
            lane.get_heading_at(0)
            lane.get_heading_at(lane.length)
        else:
            goal_type = 'u-turn'
        return goal_type
