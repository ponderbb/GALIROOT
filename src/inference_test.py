#%%
import loader
import random
from matplotlib import pyplot as plt
import cv2

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
    model.load_state_dict(torch.load('../models/baseline_b2_1.pt',map_location='cpu'))
    
    # For visualization
 
    # for idx, data in enumerate(data_load):

    #     image = data['image']
    #     keypoints = torch.mul(data['keypoints'],256)
    #     prediction = torch.mul(model(image), 256)

    #     loader.vis_keypoints(image, keypoints, prediction)
    sum_kp = torch.zeros(2)
    name = 'Batchsize 2'
    plt.figure(figsize=(64,32))
    plt.title(name)
    for idx, data in enumerate(data_load):
        plt.subplot(2,5,idx+1)
        plt.axis('off')
        image = data['image']
        keypoints = torch.mul(data['keypoints'],256)
        sum_kp = sum_kp.add(keypoints)
        prediction = torch.mul(model(image), 256)
        print(prediction)

        image_copy = loader.vis_keypoints(image, keypoints, prediction, plot=False)
        plt.imshow(image_copy)

    mean_kp = sum_kp/len(data_load)

    plt.savefig('../data/inference/{}.png'.format(name)) #TODO: this into a config

if __name__ == "__main__":
    main()