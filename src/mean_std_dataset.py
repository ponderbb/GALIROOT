
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']

if __name__ == "__main__":

    data_path ="../data/l515_lab_1410/img"

    # Train dataset
    a_transform = A.Compose([
        A.Crop(x_min=345, y_min=365, x_max=1120, y_max=1000)])

    dataset = datasets.ImageFolder(f"{data_path}", transform = transforms.Compose([Transforms(a_transform), transforms.ToTensor()]))
    loader = torch.utils.data.DataLoader(dataset,
                            batch_size=10,
                            num_workers=0,
                            shuffle=False)
    mean = 0.
    std = 0.
    nb_samples = 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(mean)
    print(std)
