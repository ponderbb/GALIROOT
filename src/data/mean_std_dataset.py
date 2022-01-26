from src import utils
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
import albumentations as A
from typing import Dict, Tuple
from tqdm import tqdm


class MeanStdDataset(Dataset):
    def __init__(self, input_path: str) -> Dict:
        self.transforms = A.Compose([A.Crop(x_min=345, y_min=365, x_max=1120, y_max=1000)])
        self.input_path = utils.list_files(input_path, ".npy")

    def __len__(self):
        return len(self.input_path)

    def __getitem__(self, idx):

        data = np.load(self.input_path[idx])

        image, depth = self._disassemble(data)
        depth = self._remap(depth, 300, 500) # FIXME: might be redundant to remap before normalizing
        depth = self._clip(depth, 0, 1)
        image = self._remap(image, image.min(), image.max()) # FIXME: might be redundant to remap before normalizing
        data_processed = self._assemble(image, depth)

        return self.transforms(image = data_processed)

    def _process_depth(self, depth):
        depth = self._remap(depth, 300, 500)
        depth = self._clip(depth, 0, 1)

    @staticmethod
    def _remap(array,  min: int, max: int):
        return (array - min) / (max - min)

    @staticmethod
    def _clip(array,  min: int, max: int):
        return array.clip(min, max)

    @staticmethod
    def _disassemble(data) -> Tuple:
        image = data[:,:,0:3]
        depth = data[:,:,3]
        return image, depth
    
    @staticmethod
    def _assemble(image, depth):
        depth = np.expand_dims(depth, axis=-1)
        return np.concatenate((image, depth), axis = -1)



def calculate_mean_std(input_path: str) -> Dict:
    dataset = MeanStdDataset(input_path)
    loader = torch.utils.data.DataLoader(dataset,
                        batch_size=10,
                        num_workers=0,
                        shuffle=False)

    rgb_mean, rgb_std, depth_mean, depth_std, nb_samples = 0, 0, 0, 0, 0
    for data in tqdm(loader):
        batch_samples = data['image'].size(0)

        rgb = data['image'][:,:,:,0:3]
        depth = data['image'][:,:,:,3].unsqueeze(-1)

        rgb = rgb.reshape(batch_samples, rgb.size(-1), -1)
        rgb_float = rgb.type(torch.FloatTensor)
        depth = depth.reshape(batch_samples, -1)
        depth_float = depth.type(torch.FloatTensor)

        rgb_mean += rgb_float.mean(2).sum(0)
        rgb_std += rgb_float.std(2).sum(0)

        depth_mean += depth_float.mean(1).sum(0)
        depth_std += depth_float.std(1).sum(0)

        nb_samples += batch_samples

    rgb_mean /= nb_samples
    rgb_std /= nb_samples
    depth_mean /= nb_samples
    depth_std /= nb_samples

    return {'rgb_mean': rgb_mean.numpy(),
            'rgb_std': rgb_std.numpy(),
            'depth_mean': depth_mean.numpy(),
            'depth_std': depth_std.numpy()}

def main():
    mean_std_dict = calculate_mean_std("data/interim")

    print('done')

if __name__ == "__main__":
    main()