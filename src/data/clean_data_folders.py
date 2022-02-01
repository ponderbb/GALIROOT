import os
import shutil


def clean_data_folders(path: str):
    try:
        shutil.rmtree(path)
    except OSError:
        print(f"Some residual remained on path {path}.")
    os.makedirs(path, exist_ok=True)
    f = open(os.path.join(path, ".gitkeep"), "w")
    f.close()


if __name__ == "__main__":

    clean_data_folders("data/interim")
    clean_data_folders("data/processed")
