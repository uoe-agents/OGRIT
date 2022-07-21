import argparse
import pickle

from igp2.data import ScenarioConfig, InDScenario
from igp2.opendrive.map import Map

from ogrit.core.base import get_occlusions_dir, get_scenarios_dir, set_working_dir
import ogrit.occlusion_detection.visualisation_tools as visualizer
from ogrit.core.data_processing import get_episode_frames


parser = argparse.ArgumentParser(description='Process the dataset')
parser.add_argument('--scenario', type=str, help='Name of scenario to process', default="bendplatz")
parser.add_argument('--episode_idx', type=int, help='Name of scenario to process', default=0)
parser.add_argument('--frame_id', type=int, help='Frame id of a specific frame to visualize', default=None)
parser.add_argument('--plot_type', type=str, help='the end results should be the same no matter the input. '
                                                  'This is used to test that the dataset '
                                                  'indeed stores the data correctly.'
                                                  'Possible arguments are:'
                                                  '     "ego" to plot all the occlusions for each ego vehicle'
                                                  '     "road" to plot the occlusions for each road '
                                                  '     "lane" to plot the occlusions for each lane',
                    default="ego")


def main():
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

    args = parser.parse_args()

    # Update the working directory to load all the file correctly.
    set_working_dir()

    # Load the map for the scenario for which we have the occlusions.
    scenario_map = Map.parse_from_opendrive(get_scenarios_dir() + f"maps/{args.scenario}.xodr")
    scenario_config = ScenarioConfig.load(get_scenarios_dir() + f"configs/{args.scenario}.json")
    scenario = InDScenario(scenario_config)
    episode = scenario.load_episode(args.episode_idx)

    # Read the json file containing the occlusions.
    with open(get_occlusions_dir() + f'/{args.scenario}_e{args.episode_idx}.p', 'rb') as f:
        occlusions = pickle.load(f)

    # Visualize the occlusions for every frame in the episode
    episode_frames = get_episode_frames(episode, exclude_parked_cars=False, exclude_bicycles=True)

    for frame_id, frame in enumerate(episode_frames):

        if args.frame_id is not None and args.frame_id != frame_id:
            continue

        frame_occlusions = occlusions[frame_id]
        for ego_id in frame_occlusions.keys():

            ego_occlusions = occlusions[frame_id][ego_id]
            if episode.agents[ego_id].parked():
                continue

            visualizer.plot_map(scenario_map, scenario_config, frame)
            visualizer.plot_ego_position(frame[ego_id].position)

            if args.plot_type == "ego":
                visualizer.plot_area(ego_occlusions["occlusions"])
            elif args.plot_type == "road":
                for road_id, _ in ego_occlusions.items():
                    if road_id == "occlusions":
                        # Here are stored all the occlusions for the ego.
                        continue
                    visualizer.plot_area(ego_occlusions[road_id]["occlusions"])
            elif args.plot_type == "lane":
                for road_id, road_occlusions in ego_occlusions.items():
                    if road_id == "occlusions":
                        # Here are stored all the occlusions for the ego.
                        continue
                    for lane_id, lane_occlusions in road_occlusions.items():
                        if lane_id == "occlusions":
                            continue
                        visualizer.plot_area(lane_occlusions)

            visualizer.show_plot()


if __name__ == "__main__":
    # Select the name of the scenario and the number of the episode you have the occlusions for.
    main()
