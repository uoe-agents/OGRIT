import os
import pathlib


def get_all_scenarios():
    return ['heckstrasse', 'bendplatz', 'frankenburg', 'neuweiler']


#### CONSTANTS ####
LSTM_PADDING_VALUE = 0
FAKE_LSTM_PADDING = -1  # used to pad the LSTM input when we have occluded frames


#### PATH CONVENTIONS ####
def get_result_file_path(scenario_name, update_hz, episode_idx):
    """ Get the path to the result file that contains the samples used by OGRIT. It assumes that unless otherwise
    specified, the standard update_hz is 25, meaning that samples are taken every 1s"""
    if update_hz != 25:
        return get_data_dir() + f'{scenario_name}_{update_hz}hz_e{episode_idx}.csv'
    else:
        return get_data_dir() + f'{scenario_name}_e{episode_idx}.csv'


def get_lstm_model_path(training_scenarios_names, input_type, update_hz, fill_occluded_frames_mode):
    """ Get the path to the LSTM model checkpoint file. """

    model_dir = os.path.join(get_lstm_dir(), f"checkpoint/")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    return os.path.join(model_dir,
                        f"{training_scenarios_names}_{input_type}_{update_hz}Hz_{fill_occluded_frames_mode}.pt")


def get_lstm_results_path(training_scenarios_names, input_type, test_scenarios_names, update_hz,
                          fill_occluded_frames_mode):
    def get_lstm_base_path(x=''):
        return get_results_dir() + f'/{test_scenarios_names}_lstm_{input_type}_on_{training_scenarios_names}_{update_hz}Hz_{fill_occluded_frames_mode}_true_goal_prob_{x}.csv'

    goal_prob_file = get_lstm_base_path()
    goal_prob_sem_file = get_lstm_base_path('sem')
    return goal_prob_file, goal_prob_sem_file


def get_lstm_dataset_path(scenario_names, input_type, split_type, update_hz, fill_occluded_frames_mode):
    return get_lstm_dir() + f"/datasets/{'_'.join(scenario_names)}_{input_type}_{split_type}_{update_hz}Hz_{fill_occluded_frames_mode}.pt"


def get_map_path(scenario_name):
    return get_scenarios_dir() + f"maps/{scenario_name}.xodr"


def get_map_configs_path(scenario_name):
    return get_scenarios_dir() + f"configs/{scenario_name}.json"


### BASIC PATH UTILS ####
def create_folders():
    # Create a folder to store the data.
    data_folder_path = get_base_dir() + '/data/'
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    # Create a folder to store the images of the decision trees.
    predictions_folder_path = get_base_dir() + '/predictions/'
    if not os.path.exists(predictions_folder_path):
        os.makedirs(predictions_folder_path)

    # Create a folder to store the images f=of the decision trees.
    img_folder_path = get_base_dir() + '/images/'
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)

    # Create a folder in which to store the occlusions.
    occlusion_folder_name = get_base_dir() + '/occlusions/'
    if not os.path.exists(occlusion_folder_name):
        os.makedirs(occlusion_folder_name)

    # Create a folder in which to store the occlusions.
    results_folder_name = get_base_dir() + '/results/'
    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)


def get_base_dir():
    return str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/'


def get_data_dir():
    return get_base_dir() + '/data/'


def get_img_dir():
    return get_base_dir() + '/images/'


def get_dt_config_dir():
    return get_base_dir() + '/ogrit/dt_config/'


def get_subset_dir():
    return get_base_dir() + '/data/significant_samples/'


def get_predictions_dir():
    return get_base_dir() + '/predictions/'


def get_occlusions_dir():
    return get_base_dir() + '/occlusions/'


def get_scenarios_dir():
    return get_base_dir() + '/scenarios/'


def get_lstm_dir():
    return get_base_dir() + '/baselines/lstm/'


def get_results_dir():
    return get_base_dir() + '/results/'


def set_working_dir():
    os.chdir(get_base_dir())


def get_igp2_results_dir():
    return get_base_dir() + '/baselines/igp2/results'


if __name__ == "__main__":
    set_working_dir()
    create_folders()
