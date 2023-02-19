# Given we just converted the trajectory files into csv, return a csv with the same columns as those
# provided in the InD dataset so that OGRIT can understand them
import numpy as np
import pandas as pd
from ogrit.core.base import get_scenarios_dir
from igp2.data.scenario import ScenarioConfig


def compute_track_lifetime(df):
    initial_frames = {}
    track_lifetimes = []
    for _, vehicle in df.iterrows():
        current_frame = vehicle["frame"]
        aid = vehicle["trackId"]
        if aid not in initial_frames:
            initial_frames[aid] = current_frame
        track_lifetimes.append(current_frame - initial_frames[aid])
    return track_lifetimes


def compute_frames(df):
    initial_frames_dict = {}
    final_frames_dict = {}
    for _, vehicle in df.iterrows():
        current_frame_nr = vehicle["frame"]
        aid = vehicle["trackId"]
        if aid not in initial_frames_dict:
            initial_frames_dict[aid] = current_frame_nr
        final_frames_dict[aid] = current_frame_nr

    return initial_frames_dict, final_frames_dict

# save episode to csv
def episode2csv(recordingID: int, trajectories: pd.DataFrame, origin:list[float]):
    recordingID = str(recordingID).zfill(2)
    """ tracks file """
    new_trajectories = pd.DataFrame(columns=["recordingId", "trackId", "frame", "trackLifetime", "xCenter", 'yCenter',
                                             "heading", "width", "length", "xVelocity", 'yVelocity',
                                             "xAcceleration", "yAcceleration", "lonAcceleration", "latAcceleration"])

    # The track_id / agent_id must be zero based.
    initial_agent_id = min(trajectories["OBJID"])
    new_trajectories["recordingId"] = [recordingID for _ in range(len(trajectories["OBJID"]))]
    new_trajectories["trackId"] = np.array(trajectories["OBJID"]) #- initial_agent_id
    new_trajectories["frame"] = [round(step/ 0.033366) for step in np.array(trajectories["TIMESTAMP"]) ]

    new_trajectories["trackLifetime"] = compute_track_lifetime(new_trajectories)
    new_trajectories["xCenter"] = np.array(trajectories["UTM_X"]) - origin[0]
    new_trajectories["yCenter"] = np.array(trajectories["UTM_Y"]) - origin[1]

    new_trajectories["heading"] = np.rad2deg(np.array(trajectories["UTM_ANGLE"]))
    new_trajectories["width"] = np.array(trajectories["WIDTH"])
    new_trajectories["length"] = np.array(trajectories["LENGTH"])

    new_trajectories["xVelocity"] = np.array(trajectories["V"]) * np.cos(np.array(trajectories["UTM_ANGLE"]))
    new_trajectories["yVelocity"] = np.array(trajectories["V"]) * np.sin(np.array(trajectories["UTM_ANGLE"]))
    new_trajectories["xAcceleration"] = np.array(trajectories["ACC"]) * np.cos(
        np.array(trajectories["UTM_ANGLE"]))
    new_trajectories["yAcceleration"] = np.array(trajectories["ACC"]) * np.sin(
        np.array(trajectories["UTM_ANGLE"]))

    new_trajectories.to_csv(data_path + "/" +recordingID + "_tracks.csv", index=False)

    """ Recording meta file """
    # From the World File, the last two numbers represent the UTM-x and -y coordinates of the top left corner
    recording_meta = pd.DataFrame(columns=["recordingId", "locationId", "frameRate", "speedLimit", "weekday",
                                           "startTime", "duration", "numTracks", "numVehicles", "numVRUs",
                                           "latLocation", "lonLocation", "xUtmOrigin", "yUtmOrigin", "orthoPxToMeter"])

    recording_meta["recordingId"] = [recordingID]
    recording_meta["locationId"] = np.nan
    recording_meta["frameRate"] = [30]
    recording_meta["weekday"] = np.nan
    recording_meta["startTime"] = [np.min(trajectories["TIMESTAMP"])]
    recording_meta["duration"] = [
        np.max(trajectories["TIMESTAMP"]) - np.min(trajectories["TIMESTAMP"])]
    recording_meta["numTracks"] = np.nan
    recording_meta["numVehicles"] = [len(new_trajectories["recordingId"].unique())]
    recording_meta["numVRUs"] = np.nan
    recording_meta["latLocation"] = np.nan
    recording_meta["lonLocation"] = np.nan
    recording_meta["xUtmOrigin"] = origin[0]
    recording_meta["yUtmOrigin"] = origin[1]
    recording_meta["orthoPxToMeter"] = np.nan

    recording_meta.to_csv(data_path + recordingID + "_recordingMeta.csv", index=False)

    """ Tracks meta file """
    tracks_meta = pd.DataFrame(columns=["recordingId", "trackId", "initialFrame", "finalFrame", "numFrames",
                                        "width", "length", "class"])

    initial_frames_dict, final_frames_dict = compute_frames(new_trajectories)

    tracks_meta["recordingId"] = [recordingID for _ in range(len(trajectories["OBJID"]))]
    tracks_meta["trackId"] = new_trajectories["trackId"]
    tracks_meta["initialFrame"] = [initial_frames_dict[aid] for aid in new_trajectories["trackId"]]
    tracks_meta["finalFrame"] = [final_frames_dict[aid] for aid in new_trajectories["trackId"]]

    # the extra frame is because the frame id is zero-based
    tracks_meta["numFrames"] = tracks_meta["finalFrame"] - tracks_meta["initialFrame"] + 1

    tracks_meta["width"] = trajectories["WIDTH"]
    tracks_meta["length"] = trajectories["LENGTH"]

    class_equivalent = {"Car": "car", "Bus": "bus", "Medium Vehicle": "truck_bus", "Heavy Vehicle": "truck_bus"}
    tracks_meta["class"] = [class_equivalent[vehicle] for vehicle in trajectories["CLASS"] ]

    tracks_meta.drop_duplicates(subset=["trackId"], inplace=True)
    tracks_meta.to_csv(data_path + recordingID + "_tracksMeta.csv", index=False)

# split one scenario into episode
def datatframe2episode(original_trajectories: pd.DataFrame, origin:list[float]):
    # each 1000 vehicles are stored in one episode
    vehicles_each_episode = 1000
    IDs = original_trajectories["OBJID"]
    start_agent_id = min(IDs)
    vehicle_num = max(IDs) - start_agent_id + 1
    recordingID = 0
    start_inx = 0
    id_last = IDs[0]
    num = 1
    for inx, id in enumerate(IDs):
        if id != id_last:
            num += 1
            id_last = id
        if num == vehicles_each_episode + 1:
            episode2csv(recordingID, original_trajectories[start_inx:inx], origin)
            start_inx = inx
            num = 1
            vehicle_num = vehicle_num - vehicles_each_episode
            recordingID += 1

            if vehicle_num < vehicles_each_episode:
                episode2csv(recordingID, original_trajectories[start_inx:], origin)
                break


if __name__ == "__main__":
    scenario_name = "rdb1"
    scenario_dir = get_scenarios_dir()
    data_path = scenario_dir + "/data/opendd/" + scenario_name + "/"
    original_trajectories = pd.read_csv(data_path + scenario_name + "_trajectories.csv")
    scenario_config = ScenarioConfig.load(scenario_dir + f"/configs/{scenario_name}.json")

    origin = [scenario_config.map_center_utm[0], scenario_config.map_center_utm[1]]

    datatframe2episode(original_trajectories, origin)



