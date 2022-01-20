import os


def list_files(directory: str, fileformat: str) -> list:
    path_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(fileformat):
                path_list.append(os.path.join(root, name))
    return sorted(path_list)
