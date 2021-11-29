from sklearn.model_selection import KFold
import loader
import torch
import utils
import numpy as np
import os
from models import models_list, loss_list
import torch.optim as optim

import wandb

def training_loop(config, device, img_list, ann_list, mask_list, train_transform, valid_transform):

    
    _, _, training = utils.first_layer_keys(config)

    kf = KFold(n_splits=training['kfold'],shuffle=True)
    num_epochs = training['epochs']
    
    all_losses={}

    for fold, (train_index, valid_index) in enumerate(kf.split(img_list)):

        early_stopping = EarlyStopping(patience=training['patience'], delta=training['delta'], path = config['folders']['out_folder']+f"temp/f{fold+1}_{training['checkpoint_name']}.pt", verbose=True)

        print('Fold {}'.format(fold + 1))

        train_img, train_ann, train_mask = utils.index_with_list(img_list, ann_list, mask_list, train_index)
        valid_img, valid_ann, valid_mask = utils.index_with_list(img_list, ann_list, mask_list, valid_index)

        train_set = loader.KeypointsDataset(config, train_ann, train_img, train_mask, train_transform, training['depth'])
        valid_set = loader.KeypointsDataset(config, valid_ann, valid_img, valid_mask, valid_transform, training['depth'])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=training['batch'])
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=training['batch'])

        model = models_list[training['model']].to(device)
        criterion = loss_list[training['loss']]
        optimizer = optim.Adam(model.parameters(), lr = training['optim_lr'], weight_decay=training['optim_wd'])
        
        history = {'train_loss': [], 'valid_loss': []}

        for epoch in range(num_epochs):

            train_loss = _train_epoch(model,device, train_loader, criterion, optimizer)
            valid_loss = _valid_epoch(model,device, valid_loader, criterion)

            print("Epoch:{}/{} AVG Training Loss:{:.5f} AVG Validation Loss:{:.5f}".format(epoch + 1,
                                                                                            num_epochs,
                                                                                            train_loss,
                                                                                            valid_loss,))

            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)

            wandb.log({'fold': fold,
                       'epoch': epoch,
                       'kfold_train_loss': train_loss,
                       'kfold_valid_loss': valid_loss})

            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        all_losses['fold{}'.format(fold+1)] = history
        
        for names, model in model.named_children():  
            try:
                # print(f"resetting {names}")
                model.reset_parameters()
            except:
                pass 

    loss_dictionary = _log_losses(config, all_losses)

    return loss_dictionary

def _train_epoch(model,device,data_loader,loss_fn,optimizer):
    train_loss=0.0
    model.train()
    for data in data_loader:
        image = data['image'].to(device)
        keypoints = data['keypoints'].squeeze().to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output,keypoints)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(data_loader.sampler)
  
def _valid_epoch(model,device,data_loader,loss_fn):
    valid_loss = 0.0
    model.eval()
    for data in data_loader:
        image = data['image'].to(device)
        keypoints = data['keypoints'].squeeze().to(device)
        output = model(image)
        loss = loss_fn(output,keypoints)
        valid_loss+=loss.item() 
    return valid_loss/len(data_loader.sampler)

def _log_losses(config, all_losses): # FIXME: this is the final boss of nightmares

    train_folds, valid_folds=[],[]

    # Calculate the average loss of each fold for all epochs # FIXME: this is a nightmare
    k = config['training']['kfold']
    for f in range(1,k+1):
        f_avg_t = np.mean(all_losses['fold{}'.format(f)]['train_loss'])
        f_avg_v = np.mean(all_losses['fold{}'.format(f)]['valid_loss'])

        wandb.log({'train_fold_avg': f_avg_t,
                   'valid_fold_avg': f_avg_v})

        train_folds.append(f_avg_t)
        valid_folds.append(f_avg_v)

    train_folds_average = np.mean(train_folds)

    valid_folds_average = np.mean(valid_folds)

    print('Performance of {} fold cross validation'.format(k))
    print("Average Training Loss: {:.5f} \t Average Test Loss: {:.5f}".format(train_folds_average,valid_folds_average))
    
    best_fold_idx = valid_folds.index(min(valid_folds))+1
    average_fold_idx = utils.closest_to_average(valid_folds_average, valid_folds)

    print(f'Best performing epoch: {best_fold_idx}\nEpoch closest to average: {average_fold_idx}')

    # log the best run separately
    for i in range(len(all_losses[f'fold{average_fold_idx}']['valid_loss'])):
        wandb.log({'best_train_loss': all_losses[f'fold{average_fold_idx}']['train_loss'][i],
                   'best_valid_loss': all_losses[f'fold{average_fold_idx}']['valid_loss'][i]})


    # Create the log file
    loss_dictionary = {"processing": config['processing'],
                       "training": config['training'],
                       "train_folds": train_folds,
                       "valid_folds": valid_folds,
                       "best": best_fold_idx,
                       "average": average_fold_idx,
                       "all_losses": all_losses}
    
    utils.dump_to_json(loss_dictionary,os.path.join(config['folders']['out_folder'],f"loss_{config['training']['checkpoint_name']}.json"))

    return loss_dictionary 
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print     

            credits: https://github.com/Bjarten/early-stopping-pytorch, https://github.com/pytorch/ignite/blob/master/ignite/handlers/early_stopping.py       
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss