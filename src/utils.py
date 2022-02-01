import os
import random
from typing import Dict

import numpy as np
import torch
import yaml


def list_files(directory: str, fileformat: str) -> list:
    path_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(fileformat):
                path_list.append(os.path.join(root, name))
    return sorted(path_list)


def load_config(path: str) -> Dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def set_seed(num: int):
    torch.manual_seed(num)
    random.seed(num)
    np.random.seed(num)

def login_wandb():
    os.system("python3 -m wandb login 80b496a88f7d314f833c5f7f62963b9c3d58a6d1")