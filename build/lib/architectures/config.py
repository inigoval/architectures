import yaml

from torch.optim import Adam, SGD

from byol_main.paths import Path_Handler

# Define paths
paths = Path_Handler()
path_dict = paths._dict()


def load_config():
    """Helper function to load yaml config file, convert to python dictionary and return."""

    # load global config
    global_path = path_dict["project"] / "config.yml"
    with open(global_path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    return config


def update_config(config):
    """Update config with values requiring initialisation of config"""
    return
