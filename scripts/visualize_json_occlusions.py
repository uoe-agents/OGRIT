import json

from igp2.data import ScenarioConfig, InDScenario
from igp2.opendrive.map import Map
from shapely.geometry import Polygon, MultiPolygon

from ogrit.core.base import get_occlusions_dir, get_scenarios_dir, set_working_dir
import ogrit.occlusion_detection.visualisation_tools as visualizer
from ogrit.core.data_processing import get_episode_frames


def main(scenario_name, episode_idx, plot_type="ego"):
    """
    Plot the occlusions from the json files.

    Args:
        scenario_name: name of the scenario for which you want to get the occlusions. Make sure you have the occlusions
                       for that scenario in the folder occlusions/
        episode_idx:   episode id of the episode for which you want the occlusions
        plot_type:     the end results should be the same no matter the input. This is used to test that the dataset
                       indeed stores the data correctly.
                       Possible arguments are:
                            "ego" to plot all the occlusions for each ego vehicle
                            "road" to plot the occlusions for each road
                            "lane" to plot the occlusions for each lane
    """

    # Update the working directory to load all the file correctly.
    set_working_dir()

    # Load the map for the scenario for which we have the occlusions.
    scenario_map = Map.parse_from_opendrive(get_scenarios_dir() + f"maps/{scenario_name}.xodr")
    scenario_config = ScenarioConfig.load(get_scenarios_dir() + f"configs/{scenario_name}.json")
    scenario = InDScenario(scenario_config)
    episode = scenario.load_episode(episode_idx)

    # Read the json file containing the occlusions.
    with open(get_occlusions_dir() + f'/{scenario_name}_e{episode_idx}.json', 'r') as f:
        occlusions = json.load(f)

    # Visualize the occlusions for every frame in the episode
    episode_frames = get_episode_frames(episode, exclude_parked_cars=False, exclude_bicycles=True)

    for frame_id, frame in enumerate(episode_frames):

        frame_occlusions = occlusions[str(frame_id)]
        for ego_id in frame_occlusions.keys():

            ego_occlusions = occlusions[str(frame_id)][str(ego_id)]
            if episode.agents[int(ego_id)].parked():
                continue

            visualizer.plot_map(scenario_map, scenario_config, frame)
            visualizer.plot_ego_position(frame[int(ego_id)].position)

            if plot_type == "ego":
                plot(ego_occlusions["occlusions"])
            elif plot_type == "road":
                for road_id, _ in ego_occlusions.items():
                    if road_id == "occlusions":
                        # Here are stored all the occlusions for the ego.
                        continue
                    plot(ego_occlusions[str(road_id)]["occlusions"])
            elif plot_type == "lane":
                for road_id, road_occlusions in ego_occlusions.items():
                    if road_id == "occlusions":
                        # Here are stored all the occlusions for the ego.
                        continue
                    for lane_id, lane_occlusions in road_occlusions.items():
                        if lane_id == "occlusions":
                            continue
                        plot(lane_occlusions)

            visualizer.show_plot()


def plot(occlusion_list):

    if len(occlusion_list) < 1:
        return

    polygons = []
    for occlusion in occlusion_list:
        polygons.append(Polygon(zip(*occlusion)))
    visualizer.plot_area(MultiPolygon(polygons))


if __name__ == "__main__":
    # Select the name of the scenario and the number of the episode you have the occlusions for.
    main("neuweiler", 1)
