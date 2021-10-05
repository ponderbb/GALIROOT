import torch 
import json
from typing import Optional
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class KeypointsDataset(Dataset):
    def __init__(self, img_dir: Path, annotations_dir: Optional[Path] = None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.images = sorted(list(self.img_dir.iterdir()))  
        self.annotations = None
        if annotations_dir:
            self.annotations = []
            for image in self.images:
                self.annotations.append(self.annotations_dir / f"{image.stem}.png.json")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        target = {}
        if self.annotations:
            with open(self.annotations[idx], "r") as f:
                annotations = json.load(f)

            if not annotations['objects']:
                raise Exception
            else: 
                target = torch.Tensor(annotations['objects'][0]['points']['exterior'])
                image = ToTensor()(Image.open(self.images[idx]))

                sample = {'image': image, 'keypoint': target}
                return sample