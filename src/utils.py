import json
import os
import matplotlib.pyplot as plt

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

def create_file(path):
    try:
        open(path, 'w').close()
    except OSError:
        print('Failed creating the file')
    else:
        print('File created')

def plot_losses(train_losses, val_losses, epoch, figure, name):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss for {}".format(name))
    plt.plot(range(epoch), val_losses,label="val")
    plt.plot(range(epoch), train_losses,label="train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(figure)

