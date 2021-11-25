from torch._C import device
import loader
import utils
from models import models_list
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import torch
import numpy as np

def inference(config, loss_dictionary, device):

    folders, processing, training = utils.first_layer_keys(config)

    ann_list, __ = utils.list_files("../data/l515_lab_1410_test/ann", processing['format_ann'])
    img_list, __ = utils.list_files("../data/l515_lab_1410_test/img", processing['format_img'])

    train_transform, valid_transform = loader.generate_transform(config)

    # index = random.sample(range(len(img_list)),5)
    index = [1,2,3,4,5]

    dataset = loader.KeypointsDataset(config, ann_list, img_list, valid_transform)
    subset = torch.utils.data.Subset(dataset, index)
    data_load = torch.utils.data.DataLoader(dataset)
    subset_load = torch.utils.data.DataLoader(subset)

    model = models_list[training['model']]
    model.eval()
    model.load_state_dict(torch.load(os.path.join(folders['out_folder'],f"{training['checkpoint_name']}.pt"),map_location=device))


    # Boxplot for distance offsets

    distance_list = []
    prediction_list = []
    ground_truth_list = []
    for data in data_load:
        image = data['image'].to(device)
        keypoints = torch.mul(data['keypoints'],256)
        prediction = torch.mul(model(image), 255)
        keypoints_np = [int(x) for x in keypoints.cpu().detach().numpy()[0][0]]
        prediction_np = [int(x) for x in prediction.cpu().detach().numpy()[0]]
        distance_list.append(utils.eucledian_dist(prediction_np, keypoints_np))
        prediction_list.append(prediction_np)
        ground_truth_list.append(keypoints_np)

    mean_list = np.mean(distance_list)
    # text = 'Average prediction\noffset: {:.3f}'.format(np.mean(distance_list))
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.figure(1, figsize=(10,5))

    fig1 = plt.subplot(1, 2, 1)
    fig1.title.set_text('Predictions on the test set')
    # fig1.text(0.55,15, text, fontsize=8, bbox=props)
    fig1.boxplot(distance_list)
    fig1.set_ylabel('Eucledian distance in pixels')

    fig2 = plt.subplot(1, 2, 2)
    fig2.title.set_text('Predicted points (x,y-axis)')
    for i,j in zip(prediction_list,ground_truth_list):
        fig2.scatter(i[0],i[1], marker = 'o', c = 'b')
        fig2.scatter(j[0],j[1], marker = 'x', c = 'k')
    fig2.set_xlim([0, 255])
    fig2.set_ylim([0, 255])
    fig2.set_xlabel('x coordinate')
    fig2.set_ylabel('y coordinate')

    plt.show()
    plt.savefig(f"../models/{training['checkpoint_name']}/box_{training['checkpoint_name']}.png")

    # Plot losses relative to the average one

    average_fold = loss_dictionary['average']
    average_fold_train = loss_dictionary['all_losses']['fold{}'.format(average_fold)]['train_loss']
    average_fold_valid = loss_dictionary['all_losses']['fold{}'.format(average_fold)]['valid_loss']

    plt.figure(2)
    plt.title("Training and validation losses")
    plt.grid()  
    plt.plot(range(len(average_fold_train)), average_fold_train, c = 'b', label="avg_train")
    plt.plot(range(len(average_fold_valid)), average_fold_valid, '--', c = 'b', label="avg_validation")
    plt.scatter(range(len(average_fold_train)), average_fold_train, marker='o', s=8 ,c = 'k', label = 'all_train')
    plt.scatter(range(len(average_fold_valid)), average_fold_valid, marker='x', s=10, c = 'k', label = 'all_validation')

    for run in range(len(loss_dictionary['train_folds'])):
        fold_train = loss_dictionary['all_losses']['fold{}'.format(run+1)]['train_loss']
        fold_valid = loss_dictionary['all_losses']['fold{}'.format(run+1)]['valid_loss']
        if len(fold_train) >= len(average_fold_train):
            plt.scatter(range(len(average_fold_train)), fold_train[:len(average_fold_train)], marker='o', s=8 ,c = 'k')
            plt.scatter(range(len(average_fold_valid)), fold_valid[:len(average_fold_valid)], marker='x', s=10, c = 'k')
        else:
            plt.scatter(range(len(fold_train)), fold_train, marker='o', s=8 ,c = 'k')
            plt.scatter(range(len(fold_train)), fold_valid, marker='x', s=10, c = 'k')
    plt.xlabel("Epochs")
    plt.xticks(range(len(average_fold_train)))
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(f"../models/{training['checkpoint_name']}/loss_graph_{training['checkpoint_name']}.png")

    plt.figure(figsize=(30,20))
    plt.title(f"Inference {training['checkpoint_name']}")
    for idx, data in enumerate(data_load):

        eucledian_dist = 0
        plt.subplot(4,5,idx+1)
        plt.axis('off')
        image = data['image'].to(device)
        keypoints = torch.mul(data['keypoints'],256)
        kp_np = [int(x) for x in keypoints.cpu().detach().numpy()[0][0]]

        prediction = torch.mul(model(image), 256)
        print(prediction)
        pred_np = [int(x) for x in prediction.cpu().detach().numpy()[0]]

        eucledian_dist = np.sqrt(np.power(pred_np[0]-kp_np[0],2)+np.power(pred_np[1]-kp_np[1],2))
        plt.text(0,300, f"Eucl. dist: {int(eucledian_dist)} pixels", color = (0,0,0), fontsize = 'xx-large')

        image_copy = utils.vis_keypoints(image, keypoints, prediction, eucledian_dist, mean_list)
        plt.imshow(image_copy)
        plt.show()

    plt.savefig(f"../models/{training['checkpoint_name']}/inf_{training['checkpoint_name']}.png")

def main():
    parser = argparse.ArgumentParser(description='configuration_file')
    parser.add_argument('-c', '--config', default='/zhome/3b/d/154066/repos/GALIROOT/config/.json', type=str, help='Configuration .json file')
    args = parser.parse_args()

    # loading configurations 
    config = utils.open_config(args.config) 
    folders, processing, training = utils.first_layer_keys(config)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    print("Running GPU.") if use_cuda else print("No GPU available.")

    loss_dictionary = utils.open_config(f"{folders['out_folder']}loss_{training['checkpoint_name']}.json")

    inference(config, loss_dictionary, device)

if __name__ == "__main__":
    main()