"""
Modified version of code from https://github.com/ika-rwth-aachen/drone-dataset-tools
"""
import io
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from descartes import PolygonPatch
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, Point

from ogrit.core.data_processing import is_target_vehicle_occluded, get_episode_frames
from ogrit.core.feature_extraction import FeatureExtractor
from matplotlib.widgets import Button, Slider
from loguru import logger
import matplotlib.image as mpimg

from ogrit.core.base import get_img_dir
from ogrit.decisiontree.dt_goal_recogniser import DecisionTreeGoalRecogniser, GeneralisedGrit, OcclusionGrit
from igp2 import VelocityTrajectory

from ogrit.occlusion_detection.visualisation_tools import plot_area_from_list


class TrackVisualizer(object):
    def __init__(self, config, tracks, static_info, meta_info, fig=None,
                 goal_recogniser=None, scenario=None, episode=None, target_agent_id=None, episode_dataset=None,
                 goal_idx=None, goal_type=None, ego_agent_id=None):
        self.config = config
        self.input_path = config["input_path"]
        self.recording_name = config["recording_name"]
        self.image_width = None
        self.image_height = None
        self.scale_down_factor = config["scale_down_factor"]
        self.skip_n_frames = config["skip_n_frames"]
        self.goal_recogniser = goal_recogniser
        self.scenario = scenario
        self.episode = episode
        self.episode_dataset = episode_dataset
        self.episode_frames = get_episode_frames(episode)
        self.currently_recording = False
        self.current_recording_idx = 0

        # agent to record in video
        self.target_agent_id = target_agent_id
        self.goal_idx = goal_idx
        self.goal_type = goal_type
        self.ego_agent_id = ego_agent_id
        self.occlusion_histories = defaultdict(list)  # dict key is agent id
        self.initial_ego_target_frame = {}

        # Get configurations
        if self.scale_down_factor % 2 != 0:
            logger.warning("Please only use even scale down factors!")

        # Tracks information
        self.tracks = tracks
        self.static_info = static_info
        self.meta_info = meta_info
        self.maximum_frames = np.max([self.static_info[track["trackId"]]["finalFrame"] for track in self.tracks])

        # Save ids for each frame
        self.ids_for_frame = {}
        for i_frame in range(self.maximum_frames):
            indices = [i_track for i_track, track in enumerate(self.tracks)
                       if
                       self.static_info[track["trackId"]]["initialFrame"] <= i_frame <= self.static_info[track["trackId"]][
                           "finalFrame"]]
            self.ids_for_frame[i_frame] = indices

        # Initialize variables
        self.current_frame = 0
        self.changed_button = False
        self.rect_map = {}
        self.plotted_objects = []
        self.track_info_figures = {}
        self.y_sign = 1

        # Create figure and axes
        if fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self.fig.set_size_inches(18, 8)
            plt.subplots_adjust(left=0.0, right=1.0, bottom=0.20, top=0.99)
        else:
            self.fig = fig
            self.ax = self.fig.gca()

        self.fig.canvas.set_window_title("Recording {}".format(self.recording_name))

        # Check whether to use the given background image
        background_image_path = self.config["background_image_path"]
        if background_image_path is not None:
            self.background_image = skimage.io.imread(background_image_path)
            self.image_height = self.background_image.shape[0]
            self.image_width = self.background_image.shape[1]
            self.ax.imshow(self.background_image)
        else:
            self.background_image = np.zeros((1700, 1700, 3), dtype=np.float64)
            self.image_height = 1700
            self.image_width = 1700
            self.ax.imshow(self.background_image)

        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # Dictionaries for the style of the different objects that are visualized
        self.rect_style = dict(fill=True, edgecolor="k", alpha=0.4, zorder=19)
        self.triangle_style = dict(facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6, zorder=19)
        self.text_style = dict(picker=True, size=8, color='k', zorder=20, ha="center")
        self.text_box_style = dict(boxstyle="round,pad=0.2", alpha=.6, ec="black", lw=0.2)
        self.track_style = dict(linewidth=1, zorder=10)
        self.centroid_style = dict(fill=True, edgecolor="black", lw=0.1, alpha=1,
                                   radius=5, zorder=30)
        self.track_style_future = dict(color="linen", linewidth=1, alpha=0.7, zorder=10)
        self.circ_stylel = dict(facecolor="g", fill=True, edgecolor="r", lw=0.1, alpha=1,
                                radius=8, zorder=30)
        self.circ_styler = dict(facecolor="b", fill=True, edgecolor="r", lw=0.1, alpha=1,
                                radius=8, zorder=30)
        self.colors = dict(car="blue", truck_bus="blue", pedestrian="red", bicycle="yellow", default="green")

        self.plot_buildings()
        # Initialize the plot with the bounding boxes of the first frame
        self.update_figure()

        ax_color = 'lightgoldenrodyellow'
        # Define axes for the widgets
        self.ax_slider = self.fig.add_axes([0.2, 0.035, 0.2, 0.04], facecolor=ax_color)  # Slider
        self.ax_button_previous2 = self.fig.add_axes([0.02, 0.035, 0.06, 0.04])
        self.ax_button_previous = self.fig.add_axes([0.09, 0.035, 0.06, 0.04])
        self.ax_button_next = self.fig.add_axes([0.44, 0.035, 0.06, 0.04])
        self.ax_button_next2 = self.fig.add_axes([0.51, 0.035, 0.06, 0.04])
        self.ax_button_play = self.fig.add_axes([0.58, 0.035, 0.06, 0.04])
        self.ax_button_stop = self.fig.add_axes([0.65, 0.035, 0.06, 0.04])
        self.ax_button_reset_ego = self.fig.add_axes([0.72, 0.035, 0.06, 0.04])
        self.ax_button_start_rec = self.fig.add_axes([0.79, 0.035, 0.06, 0.04])

        # Define the widgets
        self.frame_slider = DiscreteSlider(self.ax_slider, 'Frame', 0, self.maximum_frames-1, valinit=self.current_frame,
                                           valfmt='%s')
        self.button_previous2 = Button(self.ax_button_previous2, 'Previous x%d' % self.skip_n_frames)
        self.button_previous = Button(self.ax_button_previous, 'Previous')
        self.button_next = Button(self.ax_button_next, 'Next')
        self.button_next2 = Button(self.ax_button_next2, 'Next x%d' % self.skip_n_frames)
        self.button_play = Button(self.ax_button_play, 'Play')
        self.button_stop = Button(self.ax_button_stop, 'Stop')
        self.button_reset_ego = Button(self.ax_button_reset_ego, 'Reset Ego')
        self.button_start_rec = Button(self.ax_button_start_rec, 'Start Rec')

        # Define the callbacks for the widgets' actions
        self.frame_slider.on_changed(self.update_slider)
        self.button_previous.on_clicked(self.update_button_previous)
        self.button_next.on_clicked(self.update_button_next)
        self.button_previous2.on_clicked(self.update_button_previous2)
        self.button_next2.on_clicked(self.update_button_next2)
        self.button_play.on_clicked(self.start_play)
        self.button_stop.on_clicked(self.stop_play)
        self.button_reset_ego.on_clicked(self.reset_ego_vehicle)
        self.button_start_rec.on_clicked(self.start_rec)

        self.timer = self.fig.canvas.new_timer(interval=25*self.skip_n_frames)
        self.timer.add_callback(self.update_button_next2, self.ax)

        # Define the callbacks for the widgets' actions
        self.fig.canvas.mpl_connect('key_press_event', self.update_keypress)

        self.ax.set_autoscale_on(False)

        if self.goal_recogniser is not None:
            self.scenario.plot_goals(self.ax, self.meta_info["orthoPxToMeter"] * self.scale_down_factor, flipy=True)

    def update_keypress(self, evt):
        if evt.key == "right" and self.current_frame + self.skip_n_frames < self.maximum_frames:
            self.current_frame = self.current_frame + self.skip_n_frames
            self.trigger_update()
        elif evt.key == "left" and self.current_frame - self.skip_n_frames >= 0:
            self.current_frame = self.current_frame - self.skip_n_frames
            self.trigger_update()

    def update_slider(self, value):
        if not self.changed_button:
            self.current_frame = value
            self.trigger_update()
        self.changed_button = False

    def update_button_next(self, _):
        if self.current_frame + 1 < self.maximum_frames:
            self.current_frame = self.current_frame + 1
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning(
                "There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_next2(self, _):
        if self.current_frame + self.skip_n_frames < self.maximum_frames:
            self.current_frame = self.current_frame + self.skip_n_frames
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning("There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_previous(self, _):
        if self.current_frame - 1 >= 0:
            self.current_frame = self.current_frame - 1
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning("There are no frames available with an index lower than 1.")

    def update_button_previous2(self, _):
        if self.current_frame - self.skip_n_frames >= 0:
            self.current_frame = self.current_frame - self.skip_n_frames
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning("There are no frames available with an index lower than 1.")

    def start_rec(self, _):
        if self.currently_recording:
            print('Stopping Recording')
            self.button_start_rec.label.set_text('Start rec.')
        else:
            print('Starting Recording')
            self.current_recording_idx = 0
            self.button_start_rec.label.set_text('Stop rec.')
        self.currently_recording = not self.currently_recording

    def start_play(self, _):
        self.timer.start()

    def stop_play(self, _):
        self.timer.stop()

    def trigger_update(self):
        self.remove_patches()
        self.update_figure()
        self.update_pop_up_windows()
        self.frame_slider.update_val_external(self.current_frame)
        self.fig.canvas.draw_idle()

    def plot_occlusions(self):
        scale = 1 / (self.meta_info["orthoPxToMeter"] * self.scale_down_factor)
        plotted_objects = []
        if isinstance(self.goal_recogniser, OcclusionGrit) and self.ego_agent_id is not None:
            frame_occlusions = self.goal_recogniser.feature_extractor.occlusions[self.current_frame]
            if self.ego_agent_id in frame_occlusions:
                ego_occlusions = frame_occlusions[self.ego_agent_id]['occlusions']
                ego_occlusions_scaled = affinity.scale(ego_occlusions, xfact=scale, yfact=-scale, origin=Point([0, 0]))
                occlusion_plot = self.ax.add_patch(PolygonPatch(ego_occlusions_scaled, color='r', alpha=0.5,
                                                                fill=True, linewidth=None))
                plotted_objects.append(occlusion_plot)
        return plotted_objects

    def plot_buildings(self):
        scale = 1 / (self.meta_info["orthoPxToMeter"] * self.scale_down_factor)
        plotted_objects = []
        for building in self.scenario.config.buildings:
            poly = affinity.scale(Polygon(building), xfact=scale, yfact=-scale, origin=Point([0, 0]))
            building_plot = self.ax.add_patch(PolygonPatch(poly, color='#8a4721', alpha=0.5, fill=True, linewidth=None))
            plotted_objects.append(building_plot)
        return plotted_objects

    def update_figure(self):
        saved_tree = False
        # Plot the bounding boxes, their text annotations and direction arrow
        plotted_objects = []
        ids_for_frame = self.ids_for_frame[self.current_frame]
        for track_ind in ids_for_frame:
            track = self.tracks[track_ind]

            track_id = track["trackId"]
            static_track_information = self.static_info[track_id]
            initial_frame = static_track_information["initialFrame"]
            current_index = self.current_frame - initial_frame

            object_class = static_track_information["class"]
            is_vehicle = object_class in ["car", "truck_bus", "motorcycle"]
            bounding_box = track["bboxVis"][current_index] / self.scale_down_factor
            center_points = track["centerVis"] / self.scale_down_factor
            center_point = center_points[current_index]

            color = self.colors[object_class] if object_class in self.colors else self.colors["default"]
            if track_id == self.ego_agent_id:
                color = '#24b00e'
            elif track_id == self.target_agent_id:
                color = '#fa8602'

            if self.config["plotBoundingBoxes"] and is_vehicle:
                rect = plt.Polygon(bounding_box, True, facecolor=color, **self.rect_style)
                self.ax.add_patch(rect)
                plotted_objects.append(rect)

            if self.config["plotDirectionTriangle"] and is_vehicle:
                # Add triangles that display the direction of the cars
                triangle_factor = 0.75
                a_x = bounding_box[3, 0] + ((bounding_box[2, 0] - bounding_box[3, 0]) * triangle_factor)
                b_x = bounding_box[0, 0] + ((bounding_box[1, 0] - bounding_box[0, 0]) * triangle_factor)
                c_x = bounding_box[2, 0] + ((bounding_box[1, 0] - bounding_box[2, 0]) * 0.5)
                triangle_x_position = np.array([a_x, b_x, c_x])

                a_y = bounding_box[3, 1] + ((bounding_box[2, 1] - bounding_box[3, 1]) * triangle_factor)
                b_y = bounding_box[0, 1] + ((bounding_box[1, 1] - bounding_box[0, 1]) * triangle_factor)
                c_y = bounding_box[2, 1] + ((bounding_box[1, 1] - bounding_box[2, 1]) * 0.5)
                triangle_y_position = np.array([a_y, b_y, c_y])

                # Differentiate between vehicles that drive on the upper or lower lanes
                triangle_info = np.array([triangle_x_position, triangle_y_position])
                polygon = plt.Polygon(np.transpose(triangle_info), True, **self.triangle_style)
                self.ax.add_patch(polygon)
                plotted_objects.append(polygon)

            if self.config["plotTrackingLines"]:
                plotted_centroid = plt.Circle((center_points[current_index][0],
                                               center_points[current_index][1]),
                                              facecolor=color, **self.centroid_style)
                self.ax.add_patch(plotted_centroid)
                plotted_objects.append(plotted_centroid)
                if center_points.shape[0] > 0:
                    # Calculate the centroid of the vehicles by using the bounding box information
                    # Check track direction
                    plotted_centroids = self.ax.plot(
                        center_points[0:current_index+1][:, 0],
                        center_points[0:current_index+1][:, 1],
                        color=color, **self.track_style)
                    plotted_objects.append(plotted_centroids)
                    if self.config["plotFutureTrackingLines"]:
                        # Check track direction
                        plotted_centroids_future = self.ax.plot(
                            center_points[current_index:][:, 0],
                            center_points[current_index:][:, 1],
                            **self.track_style_future)
                        plotted_objects.append(plotted_centroids_future)

            if (self.config["showTextAnnotation"]
                    and (self.target_agent_id is None or self.target_agent_id == track_id)
                    and (not self.currently_recording or self.target_agent_id == track_id)):
                # Plot the text annotation
                annotation_text = "ID{}".format(track_id)
                if self.config["showClassLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    annotation_text += "{}".format(object_class[0])
                if self.config["showVelocityLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    current_velocity = np.sqrt(
                        track["xVelocity"][current_index] ** 2 + track["yVelocity"][current_index] ** 2) * 3.6
                    current_velocity = abs(float(current_velocity))
                    annotation_text += "{:.2f}km/h".format(current_velocity)
                if self.config["showRotationsLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    current_rotation = track["heading"][current_index]
                    annotation_text += "Deg%.2f" % current_rotation
                if self.config["showAgeLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    age = static_track_information["age"]
                    annotation_text += "Age%d/%d" % (current_index + 1, age)

                ogrit_valid = (self.ego_agent_id is not None
                               and self.ego_agent_id in ids_for_frame)

                if (self.goal_recogniser is not None
                        and len(self.episode.agents[track_id].trajectory.path) < 25 * 120
                        and object_class[0] == 'c'
                        and (not isinstance(self.goal_recogniser, OcclusionGrit) or ogrit_valid)):
                    initial_frame = static_track_information["initialFrame"]
                    frames = self.episode.frames[initial_frame:self.current_frame+1]
                    frames = [f.agents for f in frames]

                    if ogrit_valid:
                        if track_id not in self.initial_ego_target_frame:
                            self.initial_ego_target_frame[track_id] = frames[-1]
                        target_occluded = is_target_vehicle_occluded(self.current_frame,
                                                            self.goal_recogniser.feature_extractor.occlusions,
                                                            track_id, self.ego_agent_id, self.episode_frames)
                        self.occlusion_histories[track_id].append(target_occluded)

                    if isinstance(self.goal_recogniser, OcclusionGrit):
                        occlusion_histories = self.get_occlusion_history(track_id)
                        initial_frame = self.initial_ego_target_frame[track_id]
                        goal_probabilities = self.goal_recogniser.goal_probabilities(frames, track_id,
                                                    self.ego_agent_id, initial_frame, occlusion_histories)
                    else:
                        goal_probabilities = self.goal_recogniser.goal_probabilities(frames, track_id)

                    for goal_idx, prob in enumerate(goal_probabilities):
                        if prob > 0:
                            annotation_text += '\nG{}: {:.3f}'.format(goal_idx, prob)

                    # if self.target_agent_id is not None:
                    #     self.save_tree_image()
                    #     saved_tree = True

                # Differentiate between using an empty background image and using the virtual background
                target_location = (
                    center_point[0],
                    (center_point[1]))
                text_location = (
                    (center_point[0] + 45),
                    (center_point[1] - 20))
                text_patch = self.ax.annotate(annotation_text, xy=target_location, xytext=text_location,
                                              bbox={"fc": color, **self.text_box_style}, **self.text_style)
                plotted_objects.append(text_patch)

        plotted_objects.extend(self.plot_occlusions())

        # Add listener to figure
        self.fig.canvas.mpl_connect('pick_event', self.display_features_on_click)
        # Save the plotted objects in a list
        self.plotted_objects = plotted_objects

        if self.currently_recording:
            self.save_road_image()

    def get_occlusion_history(self, track_id):
        occlusion_histories = self.occlusion_histories[track_id]
        fps = 25
        final_idx = len(occlusion_histories) - 1
        occlusion_histories = occlusion_histories[final_idx % fps:fps:final_idx + 1]
        return occlusion_histories

    def save_tree_image(self):
        goal_idx = self.goal_idx
        goal_type = self.goal_type
        if isinstance(self.goal_recogniser, GeneralisedGrit):
            pydot_tree = self.goal_recogniser.decision_trees[goal_type].pydot_tree()
        else:
            pydot_tree = self.goal_recogniser.decision_trees[goal_idx][goal_type].pydot_tree()

        directory = get_img_dir() + '/video_tree'
        if not os.path.exists(directory):
            os.makedirs(directory)

        pydot_tree.write_png(directory + '/{}.png'.format(self.current_frame))

    def save_road_image(self):

        directory = get_img_dir() + f'/video_{self.ego_agent_id}_{self.target_agent_id}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig = plt.gcf()
        extent = self.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(directory + f'/img{self.current_recording_idx}.png', bbox_inches=extent, pad_inches=0)
        self.current_recording_idx += 1

    def display_features_on_click(self, event):
        artist = event.artist
        text_value = artist._text
        if "ID" not in text_value:
            return

        track_id_string = text_value[:text_value.index("|")]
        track_id = int(track_id_string[2:])

        if self.ego_agent_id is None:
            self.ego_agent_id = track_id
            self.occlusion_histories = defaultdict(list)  # dict key is agent id
            self.initial_ego_target_frame = {}
            print(f'Set ego agent to {track_id}')
            return
        print(f'Showing details for target agent {track_id} with ego agent {self.ego_agent_id}')

        self.target_agent_id = track_id
        print(f'Set target agent to {track_id}')

        # do goal inference to get correct higlighting on tree
        static_track_information = self.static_info[track_id]
        initial_frame = static_track_information["initialFrame"]
        frames = self.episode.frames[initial_frame:self.current_frame + 1]
        frames = [f.agents for f in frames]

        state_history = [f[track_id] for f in frames]
        trajectory = VelocityTrajectory.from_agent_states(state_history)
        typed_goals = self.goal_recogniser.feature_extractor.get_typed_goals(trajectory, self.scenario.config.goals,
                                                                             self.scenario.config.goal_threshold)

        #goal_probabilities = self.goal_recogniser.goal_probabilities(frames, track_id)

        #select ego agent here
        agent_data = self.episode_dataset.loc[(self.episode_dataset.agent_id == track_id)
                                              & (self.ego_agent_id==self.episode_dataset.ego_agent_id)]
        goal_idxes = agent_data.possible_goal.unique()

        features = list(FeatureExtractor.feature_names.keys()) + FeatureExtractor.indicator_features

        for goal_idx in goal_idxes:
            goal_data = agent_data.loc[agent_data.possible_goal==goal_idx]
            goal_type = goal_data.goal_type.iloc[0]
            fig = plt.figure(np.random.randint(0, 5000, 1))
            fig.canvas.mpl_connect('close_event', lambda evt: self.close_track_info_figure(evt, track_id))
            fig.canvas.mpl_connect('resize_event', lambda evt: fig.tight_layout())
            fig.set_size_inches(12, 7)
            fig.canvas.set_window_title("Recording {}, Track {}, Goal {}".format(self.recording_name,
                                                                                 track_id, goal_idx))

            for feature_idx, feature in enumerate(features):
                feature_data = goal_data[['frame_id', feature]]
                sub_plot = plt.subplot(4, 4, feature_idx + 1, title=feature)
                plt.plot(feature_data.frame_id, feature_data[feature])
                borders = [feature_data[feature].min() - 1, feature_data[feature].max() + 1]
                plt.plot([self.current_frame, self.current_frame], borders, "--r")
                plt.xlabel('frame')
                sub_plot.grid(True)

            if isinstance(self.goal_recogniser, OcclusionGrit):
                self.goal_recogniser.goal_likelihood(frames, typed_goals[goal_idx], track_id, self.ego_agent_id,
                                                     self.initial_ego_target_frame[track_id],
                                                     self.get_occlusion_history(track_id))
                pydot_tree = self.goal_recogniser.decision_trees[goal_type].pydot_tree()
            elif isinstance(self.goal_recogniser, GeneralisedGrit):
                self.goal_recogniser.goal_likelihood(frames, typed_goals[goal_idx], track_id)
                pydot_tree = self.goal_recogniser.decision_trees[goal_type].pydot_tree()
            else:
                self.goal_recogniser.goal_likelihood(goal_idx, frames, typed_goals[goal_idx], track_id)
                pydot_tree = self.goal_recogniser.decision_trees[goal_idx][goal_type].pydot_tree()

            png_str = pydot_tree.create_png(prog='dot')
            sio = io.BytesIO()
            sio.write(png_str)
            sio.seek(0)
            img = mpimg.imread(sio)

            # plot the image
            fig = plt.figure()
            imgplot = plt.imshow(img, aspect='equal')
            title = f'G{goal_idx} {goal_type}'
            fig.canvas.set_window_title(title)
            plt.title(title)

        plt.show()
        self.ego_agent_id = None

    def reset_ego_vehicle(self, event):
        self.ego_agent_id = None
        self.occlusion_histories = defaultdict(list)  # dict key is agent id
        self.initial_ego_target_frame = {}
        print('Resetting ego vehicle')

    def on_click(self, event):
        artist = event.artist
        text_value = artist._text
        if "ID" not in text_value:
            return

        try:
            track_id_string = text_value[:text_value.index("|")]
            track_id = int(track_id_string[2:])
            track = None
            for track in self.tracks:
                if track["trackId"] == track_id:
                    track = track
                    break
            if track is None:
                logger.error("No track with the ID {} was found. Nothing to show.".format(track_id))
                return
            static_information = self.static_info[track_id]

            # Get information of the selected track
            centroids = track["center"]
            rotations = track["heading"]
            centroids = np.transpose(centroids)
            initial_frame = static_information["initialFrame"]
            final_frame = static_information["finalFrame"]
            x_limits = [initial_frame, final_frame]
            track_frames = np.linspace(initial_frame, final_frame, centroids.shape[1], dtype=np.int64)

            # Create a new figure that pops up
            fig = plt.figure(np.random.randint(0, 5000, 1))
            fig.canvas.mpl_connect('close_event', lambda evt: self.close_track_info_figure(evt, track_id))
            fig.canvas.mpl_connect('resize_event', lambda evt: fig.tight_layout())
            fig.set_size_inches(12, 7)
            fig.canvas.set_window_title("Recording {}, Track {} ({})".format(self.recording_name,
                                                                             track_id, static_information["class"]))

            borders_list = []
            subplot_list = []

            key_check_list = ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
            counter = 3
            for check_key in key_check_list:
                if check_key in track and track[check_key] is not None:
                    counter = counter + 1
            if 3 < counter <= 6:
                subplot_index = 321
            elif 6 < counter <= 9:
                subplot_index = 331
            else:
                subplot_index = 311

            # ---------- X POSITION ----------
            sub_plot = plt.subplot(subplot_index, title="X-Position [m]")
            subplot_list.append(sub_plot)
            x_positions = centroids[0, :]
            borders = [np.amin(x_positions), np.amax(x_positions)]
            plt.plot(track_frames, x_positions)
            red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
            self.plotted_objects.append(red_line)
            borders_list.append(borders)
            plt.xlim(x_limits)
            plt.ylim(borders)
            sub_plot.grid(True)
            plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- Y POSITION ----------
            sub_plot = plt.subplot(subplot_index, title="Y-Position [m]")
            subplot_list.append(sub_plot)
            y_positions = centroids[1, :]
            borders = [np.amin(y_positions), np.amax(y_positions)]
            plt.plot(track_frames, y_positions)
            red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
            self.plotted_objects.append(red_line)
            borders_list.append(borders)
            plt.xlim(x_limits)
            plt.ylim(borders)
            sub_plot.grid(True)
            plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- HEADING ----------
            sub_plot = plt.subplot(subplot_index, title="Heading [deg]")
            subplot_list.append(sub_plot)
            borders = [-10, 400]
            plt.plot(track_frames, np.unwrap(rotations, discont=360))
            red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
            self.plotted_objects.append(red_line)
            borders_list.append(borders)
            plt.xlim(x_limits)
            plt.ylim(borders)
            sub_plot.grid(True)
            plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "xVelocity" ----------
            if "xVelocity" in track and track["xVelocity"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="X-Velocity [m/s]")
                subplot_list.append(sub_plot)
                x_velocity = track["xVelocity"]
                borders = [np.amin(x_velocity), np.amax(x_velocity)]
                plt.plot(track_frames, x_velocity)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "yVelocity" ----------
            if "yVelocity" in track and track["yVelocity"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Y-Velocity [m/s]")
                subplot_list.append(sub_plot)
                y_velocity = track["yVelocity"]
                borders = [np.amin(y_velocity), np.amax(y_velocity)]
                plt.plot(track_frames, y_velocity)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "lonVelocity" ----------
            if "lonVelocity" in track and track["lonVelocity"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Longitudinal-Velocity [m/s]")
                subplot_list.append(sub_plot)
                lon_velocity = track["lonVelocity"]
                borders = [np.amin(lon_velocity), np.amax(lon_velocity)]
                plt.plot(track_frames, lon_velocity)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "XAcceleration" ----------
            if "xAcceleration" in track and track["xAcceleration"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="X-Acceleration [m/s^2]")
                subplot_list.append(sub_plot)
                x_acc = track["xAcceleration"]
                borders = [np.amin(x_acc), np.amax(x_acc)]
                plt.plot(track_frames, x_acc)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "yAcceleration" ----------
            if "yAcceleration" in track and track["yAcceleration"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Y-Acceleration [m/s^2]")
                subplot_list.append(sub_plot)
                y_acc = track["yAcceleration"]
                borders = [np.amin(y_acc), np.amax(y_acc)]
                plt.plot(track_frames, y_acc)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "yAcceleration" ----------
            if "lonAcceleration" in track and track["lonAcceleration"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Longitudinal-Acceleration [m/s^2]")
                subplot_list.append(sub_plot)
                lon_acc = track["lonAcceleration"]
                borders = [np.amin(lon_acc), np.amax(lon_acc)]
                plt.plot(track_frames, lon_acc)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')

            self.track_info_figures[track_id] = {"main_figure": fig,
                                                 "borders": borders_list,
                                                 "subplots": subplot_list}
            plt.show()
        except Exception:
            logger.exception("An exception occured trying to display track information")
            return

    def close_track_info_figure(self, evt, track_id):
        if track_id in self.track_info_figures:
            self.track_info_figures[track_id]["main_figure"].canvas.mpl_disconnect('close_event')
            self.track_info_figures.pop(track_id)

    def get_figure(self):
        return self.fig

    def remove_patches(self):
        self.fig.canvas.mpl_disconnect('pick_event')
        for figure_object in self.plotted_objects:
            if isinstance(figure_object, list):
                figure_object[0].remove()
            else:
                figure_object.remove()
        self.plotted_objects = []

    def update_pop_up_windows(self):
        for track_id, track_map in self.track_info_figures.items():
            borders = track_map["borders"]
            subplots = track_map["subplots"]
            for subplot_index, subplot_figure in enumerate(subplots):
                new_line = subplot_figure.plot([self.current_frame, self.current_frame], borders[subplot_index], "--r")
                self.plotted_objects.append(new_line)
            track_map["main_figure"].canvas.draw_idle()

    @staticmethod
    @logger.catch(reraise=True)
    def show():
        plt.show()


class DiscreteSlider(Slider):
    def __init__(self, *args, **kwargs):
        self.inc = kwargs.pop('increment', 1)
        self.valfmt = '%s'
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        if self.val != val:
            discrete_val = int(int(val / self.inc) * self.inc)
            xy = self.poly.xy
            xy[2] = discrete_val, 1
            xy[3] = discrete_val, 0
            self.poly.xy = xy
            self.valtext.set_text(self.valfmt % discrete_val)
            if self.drawon:
                self.ax.figure.canvas.draw()
            self.val = val
            if not self.eventson:
                return
            for cid, func in self.observers.items():
                func(discrete_val)

    def update_val_external(self, val):
        self.set_val(val)