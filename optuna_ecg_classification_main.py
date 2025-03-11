import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
import numpy as np
from torch.utils.data import DataLoader,Dataset
from bs4 import BeautifulSoup
import numpy as np
import glob
import os.path
import sys, os
import argparse
import yaml
import random
import utils as ut
import pandas as pd
import argparse
from tqdm import tqdm
import torch.nn.init as init
import torchvision
import optuna
from optuna.trial import TrialState


###########################################################
# Auxiliar functions
###########################################################

#Set the same seed for all the generators
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


#Routine used in training loop and validation loop
#Get's the network predictions and the loss
def train_1_epoch(model, optimizer, criterion, train_loader, device):
    
    #Toggle training mode
    model.train()

    #Vector to save all the batches loss
    epoch_loss = []
   
    #Run over all the training test
    for batch in train_loader:

        
        ecg, labels = batch
        
        #Pass arguments to gpu
        ecg = ecg.to(device).float()
        labels = labels.to(device)
        
        #Predictions
        out = model(ecg)
        
        #to gpu
        out = out.to(device)
        
        #Calculate loss
        if criterion == "focal_loss":
            loss = torchvision.ops.sigmoid_focal_loss(out, labels, alpha = config['focal_loss_alpha'], gamma = config['focal_loss_gamma'], reduction="mean")
        else:
            loss = criterion(out, labels) 
        
        #BackPropagation
        optimizer.zero_grad() # zero the gradient buffers
        loss.backward()
        optimizer.step() # Does the update

        #Add batch loss to
        epoch_loss.append(loss.item())

        
    return np.mean(epoch_loss)

#Routine used in training loop and validation loop
#Get's the network predictions and the loss
def evaluate(model, criterion, eval_loader, class_threshold, device):
    
    #Toggle training mode
    model.eval()

    #We do not need gradients
    with torch.no_grad():

        #Vector to save all the batches loss
        eval_loss = []
        all_preds = []
        all_probs = []
        all_labels = []

        #Run over all the training test
        for batch in tqdm(eval_loader):

            ecg, labels = batch

            #Pass arguments to gpu
            ecg = ecg.to(device).float()
            labels = labels.to(device)
            
            #Predictions
            out = model(ecg)
            
            #to gpu
            out = out.to(device)
            
            if criterion == "focal_loss":
                loss = torchvision.ops.sigmoid_focal_loss(out, labels, alpha = config['focal_loss_alpha'], gamma = config['focal_loss_gamma'], reduction="mean")
            else:
                loss = criterion(out, labels)

            eval_loss.append(loss.item())
            
            
            #Get the probabilities
            out_probs = torch.sigmoid(out)
            predictions = (out_probs > class_threshold).int()
            

            all_preds.append(predictions.cpu())
            all_labels.append(labels.int().cpu())
            all_probs.append(out_probs.cpu())

    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()


    #Return predictions and 
    return np.mean(eval_loss), all_preds, all_labels, all_probs

###############################
## Main function
###############################

#Optuna optimization function

def objective(trial, config, args):

    #Setting seed in order to replicate experiments
    set_seed(config['seed'])

    #Defining optuna parameters to optimize
    config['lr'] = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    config['norm'] = trial.suggest_categorical("normalization", ["minmax","zscore", "rscal", "logscal", "l2"])
    
    config['rnn_layers'] = trial.suggest_categorical("rnn_layers", [1, 2, 4])
    config['rnn_hidden_size'] = trial.suggest_categorical("rnn_hidden_size", [128, 256, 512])
    config['rnn_dropout'] = trial.suggest_categorical("rnn_dropout", [0, 0.1, 0.3])
    config['bidirectional'] = trial.suggest_categorical("bidirectional", [True, False])
    

    print('Starting...')
       

    #prepare statements file, classes file
    print("Processing...")

    #pick dataset, process it and returns it as well as each set we will use
    dataset, train_set, val_set, test_set = ut.pick_dataset(config)
    
    print('Creating Dataloaders...')

    #train_loader
    train_loader = DataLoader(
                dataset = train_set, 
                batch_size = config['batch_size'], 
                shuffle =True, 
                drop_last=True,
                num_workers = config['n_workers'],
                pin_memory = config['pin_memory'], 
                prefetch_factor = config['prefetch_factor'], 
                )
    
    val_loader = DataLoader(
                            val_set, 
                            batch_size = config['batch_size'], 
                            shuffle = False,
                            drop_last=False, 
                            num_workers = config['n_workers'],
                            pin_memory = config['pin_memory'], 
                            prefetch_factor = config['prefetch_factor'], 
                            )
    

    test_loader = DataLoader(
                            test_set, 
                            batch_size = config['batch_size'], 
                            shuffle = False,
                            drop_last=False, 
                            num_workers = config['n_workers'],
                            pin_memory = config['pin_memory'], 
                            prefetch_factor = config['prefetch_factor'], 
                            )

    print('Dataloaders created')


    print('Looking for these classes:')
    print(dataset.get_dataset_targets())
    #number of output classes
    n_classes = len(dataset.get_dataset_targets())

    #Defining the model
    model = ut.choose_model(config["model_type"], config["submodel"], n_classes, config)
    #Initialization of the model
    model.apply(ut.init_weights)
    
    # Transfer model to gpu
    if torch.cuda.is_available():
        model.to(args.device)
    
    #Loss and optimizer definitions
    optimizer = torch.optim.AdamW(model.parameters(), config["lr"])

    if config['scheduler'] == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if config['pos_weight'] != 'none' and config['criterion'] == "bce_logits":
        print('BCE with logits and pos weight')
        pos_weight = torch.tensor(config['pos_weight'], dtype=torch.float32).to(args.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    elif config['criterion'] == "focal_loss":
        print('Focal Loss')
        criterion = "focal_loss"

    else:
        print('BCE with logits and without pos weight')
        criterion = nn.BCEWithLogitsLoss()

    print('Dataset length: ', len(train_loader.dataset))

    #Loop over epochs
    for epoch in tqdm(range(config['epochs'])):

        #Loss_vector
        training_loss = []
        eval_loss = []

        #Train for 1 epoch
        epoch_loss = train_1_epoch(model, optimizer, criterion, train_loader, args.device)
        training_loss.append(epoch_loss)

        print('Training Loss: %.6f'%epoch_loss)
        #Evaluate the model on the validation set
        # val loss: mean loss during eval
        # predictions: Network output round to 0 or 1. n_samples x n_classes
        # labels: targets n_samples x n_classes
        val_loss, predictions, labels, probs = evaluate(model, criterion, test_loader,config['class_threshold'], args.device)

        print('Eval loss: %.6f'% val_loss)
        #Receives predictions, targets, probabilities, the tags for each class and the mode - print or return
        metrics = ut.get_metrics(predictions, labels, probs, dataset.get_dataset_targets(), 'print')


        #Update scheduler if one was chosen
        if config['scheduler'] != 'none':
            scheduler.step()

        #Make the loop check what we are optimizing
        #trial.report(metrics["f1_score_overall"], epoch)
        trial.report(metrics['f1_score_overall'], epoch)
        

        # Handle pruning based on the intermediate value.
        #Only available in single objective mode
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return metrics['f1_score_overall']#metrics["f1_score_overall"]#, metrics['gmean_overall'], metrics['ap_macro']

if __name__ == '__main__':

    #Get hyperparameters
    args = argparse.Namespace(
        #yaml="/mnt/l/git_repositories/ECG_done_right/hyperparameters.yml",
        yaml="path_to_hyperparameters.yml",
        device="cuda:0"
    )

    #open yaml args
    with open(args.yaml, "r") as f:
        config = yaml.safe_load(f)

    study = optuna.create_study(directions=["maximize"])#, "maximize", "maximize"])
    
    #Runs for 30 trials or at most 8hours
    study.optimize(lambda trial: objective(trial, config, args), n_trials=100, timeout = 35000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    #Finish
    print(config['model_type'], config['submodel'])
    print('Run finished!')
    
