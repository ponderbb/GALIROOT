#%%
import loader
import random

from models import SelfNet

import torch



def main(): # TODO: clean up inference and creat config

    config = '/zhome/3b/d/154066/repos/GALIROOT/config/gbar_1_dataset.json'
    dataset = loader.KeypointsDataset(config, transform=True)
    index = random.sample(range(111),10)
    # print(index)

    subset1 = torch.utils.data.Subset(dataset, index)
    data_load = torch.utils.data.DataLoader(
        subset1
    )
    
    # load model
    model = SelfNet()
    model.eval()
    model.load_state_dict(torch.load('/zhome/3b/d/154066/repos/GALIROOT/models/baseline_b2_sf0_wd.pt',map_location='cpu'))
    
    # For visualization

    for idx, data in enumerate(data_load):
        image = data['image']
        keypoints = torch.mul(data['keypoints'],256)
        pred_raw = model(image)
        prediction = torch.mul(pred_raw, 256)
        loader.vis_keypoints(image, keypoints, prediction)

if __name__ == "__main__":
    main()