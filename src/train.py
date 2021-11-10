from sklearn.model_selection import KFold
import loader
import torch
import utils
import numpy as np
import os
from models import models_list, loss_list, EucledianLoss
import torch.optim as optim

def training_loop(config, device, img_list, ann_list, train_transform, valid_transform):

    
    _, _, training = utils.first_layer_keys(config)

    kf = KFold(n_splits=training['kfold'],shuffle=True)
    num_epochs = training['epochs']
    min_valid_loss = np.inf

    foldperf={}
    epochperf ={"train_epochs": [0]*training['epochs'], "valid_epochs": [0]*training['epochs']}

    for fold, (train_index, valid_index) in enumerate(kf.split(img_list)):

        print('Fold {}'.format(fold + 1))

        train_img, train_ann = utils.index_with_list(img_list, ann_list, train_index)
        valid_img, valid_ann = utils.index_with_list(img_list,ann_list,valid_index)

        train_set = loader.KeypointsDataset(config, train_ann, train_img, train_transform)
        valid_set = loader.KeypointsDataset(config, valid_ann, valid_img, valid_transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=training['batch'])
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=training['batch'])

        net = models_list[training['model']].to(device)
        criterion = loss_list[training['loss']]
        optimizer = optim.Adam(net.parameters(), lr = training['optim_lr'], weight_decay=training['optim_wd'])
        
        history = {'train_loss': [], 'valid_loss': []}

        for epoch in range(num_epochs):

            train_loss = _train_epoch(net,device, train_loader, criterion, optimizer)
            valid_loss = _valid_epoch(net,device, valid_loader, criterion)

            print("Epoch:{}/{} AVG Training Loss:{:.5f} AVG Validation Loss:{:.5f}".format(epoch + 1,
                                                                                            num_epochs,
                                                                                            train_loss,
                                                                                            valid_loss,))

            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)



            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model') # valid_loss -> average validation loss
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(net.state_dict(), config['folders']['out_folder']+f"bv_{training['checkpoint_name']}.pt")

        foldperf['fold{}'.format(fold+1)] = history

        epochperf['train_epochs'] = [sum(e) for e in zip(epochperf['train_epochs'],history['train_loss'])]
        epochperf['valid_epochs'] = [sum(e) for e in zip(epochperf['valid_epochs'],history['valid_loss'])]

        torch.save(net.state_dict(), config['folders']['out_folder']+f"{training['checkpoint_name']}.pt")
        
        for names, net in net.named_children():  
            try:
                # print(f"resetting {names}")
                net.reset_parameters()
            except:
                pass 

    loss_dictionary = _log_losses(config, foldperf, epochperf)

    return loss_dictionary



def _train_epoch(net,device,data_loader,loss_fn,optimizer):
    train_loss=0.0
    net.train()
    for data in data_loader:
        image = data['image'].to(device)
        keypoints = data['keypoints'].squeeze().to(device)
        optimizer.zero_grad()
        output = net(image)
        loss = loss_fn(output,keypoints)
        # loss = EucledianLoss(output,keypoints,device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(data_loader.sampler)
  
def _valid_epoch(net,device,data_loader,loss_fn):
    valid_loss = 0.0
    net.eval()
    for data in data_loader:
        image = data['image'].to(device)
        keypoints = data['keypoints'].squeeze().to(device)
        output = net(image)
        loss = loss_fn(output,keypoints)
        # loss=EucledianLoss(output,keypoints, device)
        valid_loss+=loss.item() 
    return valid_loss/len(data_loader.sampler)

def _log_losses(config, foldperf, epochperf):

    train_folds, valid_folds, train_epochs, valid_epochs=[],[],[],[]

    # Calculate the average loss of each fold for all epochs
    k = config['training']['kfold']
    for f in range(1,k+1):
        train_folds.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        valid_folds.append(np.mean(foldperf['fold{}'.format(f)]['valid_loss']))

    print('Performance of {} fold cross validation'.format(k))
    print("Average Training Loss: {:.5f} \t Average Test Loss: {:.5f}".format(np.mean(train_folds),np.mean(valid_folds)))

    # Calculate the average loss of each epoch over folds
    num_epochs = config['training']['epochs']
    train_epochs = [e/num_epochs for e in epochperf['train_epochs']]
    valid_epochs = [e/num_epochs for e in epochperf['valid_epochs']]

    # Create the log file

    loss_dictionary = {"processing": config['processing'], "training": config['training'], "train_folds": train_folds, "valid_folds": valid_folds, "train_epochs": train_epochs, "valid_epochs": valid_epochs}
    
    utils.dump_to_json(loss_dictionary,os.path.join(config['folders']['out_folder'],f"loss_{config['training']['checkpoint_name']}.json"))

    return loss_dictionary # FIXME: how do you get over pasing the same shit from the function inception???