import json
import os

def open_config(path):
    with open(path) as j:
        config_json = json.load(j)
    return config_json

def list_files(directory, fileformat):
    path_list = []
    name_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(fileformat):
                path_list.append(os.path.join(root, name))
                name_list.append(name)
    return sorted(path_list), sorted(name_list)