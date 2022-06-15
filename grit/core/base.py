import pathlib


def get_base_dir():
    return str(pathlib.Path(__file__).parent.parent.parent.absolute())


def get_data_dir():
    return get_base_dir() + '/data/'


def get_img_dir():
    return get_base_dir() + '/images/'


def get_dt_config_dir():
    return get_base_dir() + '/grit/dt_config/'


def get_subset_dir():
    return get_base_dir() + '/data/significant_samples/'


def get_predictions_dir():
    return get_base_dir() + '/predictions/'