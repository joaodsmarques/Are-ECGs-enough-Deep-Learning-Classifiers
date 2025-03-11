# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:22:21 2023

@author: Utilizador
"""

import numpy as np
import pywt
from scipy.signal import butter,filtfilt
import pandas as pd
import ast
import wfdb
import random
import torch
import torch.nn as nn
import bwr
import data_augmentation as dtaug
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_auc_score, average_precision_score
from scipy.signal import butter, sosfilt
from torch.utils.data import DataLoader,Dataset
import sys
import os
from bs4 import BeautifulSoup

#All models we want to import
import torch.nn.init as init
from models import (AlexNet, 
                    vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn,vgg19_bn, vgg19 ,
                    resnet18, resnet34, resnet50, resnet101, resnet152,
                    AttResNet, ECG_RNN, CRNN, ECG_Transformer_Encoder, ResTransformer)


##################################################
# Plot ECG leads
################################################################
def plot_ecg(data, leads_to_plot=None, tags = ['I', 'II', 'III'], sample_rate = 500):
    """
    Plots ECG data from a PyTorch tensor.
    
    Args:
        data: 12 lead ecg in shape (length_lead x n_leads)
        leads_to_plot: Which ones we want to be plotted
        tags: vector with all the possible lead names for this type of ecg
        sample_rate (int): Sampling rate in Hz (default is 500 Hz).
    """

    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data must be a PyTorch tensor.")
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("Input tensor must have shape [n_samples, n_leads].")
    

    n_leads, n_samples = data.shape
    
    # Validate lead indices
    for lead in leads_to_plot:
        if lead not in tags:
            raise ValueError(f"Lead index {lead} is out of range for data tag leads.")
    
    # Create time axis
    time = torch.arange(n_samples) / sample_rate
    
    # Plot each selected lead
    plt.figure(figsize=(12, len(leads_to_plot) * 2))

    for i, lead_name in enumerate(leads_to_plot):
        plt.subplot(len(leads_to_plot), 1, i + 1)

        #Check the index correspondent to the lead we are selecting
        for lead_index in range(len(tags)):
            if tags[lead_index] == lead_name:
                break
        plt.plot(time, data[lead_index], label=f"Lead {lead_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

########################################################################################

# Defining global important variables for processing the signal

sampling_rate = 500

# Filter requirements. - It has to accomplish the nyquist theorem fs > 2fsignal
fs = 500.0       # sample rate, Hz
cutoff = 50.0     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = 5000 # total number of samples

band_pass_filter = butter(2, [1, 45], 'bandpass', fs=100, output='sos') #bandpasss filter for preprocessing

##########################################################################################

#Butterworth Filter aplliance
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff/nyq
    
    #Get filter coefficients
    b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
    y = filtfilt (b,a, data)
    
    return y


def calc_baseline(signal):
    """
    Calculate the baseline of signal.

    Args:
        signal (numpy 1d array): signal whose baseline should be calculated


    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]

 

############################################################################
# Get the tags for the leads of the ECG
############################################################################
def get_ecg_labels(dataset):

    if dataset == "ptbxl":
        return ['I','II','III','AVR','AVL','AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    elif dataset == "hsm":
        return ['I','II','III','aVR','aVL','aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    elif dataset == "cpsc18":
        return ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    else:
        raise Exception("Not yet implemented! Only 'ptbxl' and 'hsm' are available.")
    

#Loads the auxiliar documents and prepares the data_files containting the information needed for the dataloader
class prepare_ptbxl_dataset():
  
    def __init__(self, path_to_dataset = "", labels_set = "diagnostic", labels_subset = "class"):
        
        '''
            Function to organize the labels and data of our dataset

            path_to_dataset: path to the scp_statements.csv file and ptbxl_database.csv file
            labels_set: what labels are we using (diagnostic, form, rythm or all)
            labels_subset: if the labers chosen are diagnostic, we can choose superclass, all or subclass
        '''     
        self.target_labels = ""
        self.dataset = None
        self.labels_set = labels_set
        self.labels_subset = labels_subset

        # Load scp_statements.csv for diagnostic aggregation
        self.scp_statements = pd.read_csv(path_to_dataset +'scp_statements.csv', index_col=0)
        #Drop unwanted columns
        self.scp_statements.drop(columns = ["SCP-ECG Statement Description", 
                                            "AHA code", 
                                            "aECG REFID", 
                                            "CDISC Code",
                                            "DICOM Code"], 
                                            inplace = True)
        
        #Get just the labels set we want to filter from all the labels
        if labels_set == "diagnostic":
            self.scp_statements = self.scp_statements[self.scp_statements.diagnostic == 1]
            
            #All Diagnostic labels
            if self.labels_subset == "all":
                self.target_labels = list(set(self.scp_statements.index))

            #Diagnostic Superclass labels
            elif self.labels_subset == "superclass":
                self.target_labels = list(set(self.scp_statements.diagnostic_class))

            #Diagnostic subclass labels
            elif self.labels_subset == "subclass":
                self.target_labels = list(set(self.scp_statements.diagnostic_subclass))

            else:
                raise Exception("Invalid subclass set. Choose 'all', 'superclass' or 'subclass'.")

        elif labels_set == "form":
            self.scp_statements = self.scp_statements[self.scp_statements.form == 1]
            self.target_labels = list(set(self.scp_statements.index))

        elif labels_set == "rhythm":
            self.scp_statements = self.scp_statements[self.scp_statements.rhythm == 1]
            self.target_labels = list(set(self.scp_statements.index))

        #the dataframe is kept the same as all labels are kept
        elif labels_set == "all":
            self.scp_statements = self.scp_statements
            self.target_labels = list(set(self.scp_statements.index))

        else:
            raise Exception("Invalid selection of labels. Only 'diagnostic', 'form', 'rhythm' or 'all' are valid selections.")
        
        # load and convert annotation data - csv with all ECG data
        self.database = pd.read_csv(path_to_dataset + 'ptbxl_database.csv', index_col='ecg_id')

        #Sort target labels
        self.target_labels = sorted(self.target_labels)
    #Aggregate classes by example
    def aggregate_diagnostic(self, y_dic):
        tmp = []
        
        #For each class in a specific ecg
        for scp_code, value in y_dic.items():

            #If the key exists in the classes we are filtering
            if scp_code in self.scp_statements.index:
                
                #if the classes are diagnostic
                if self.labels_set == "diagnostic":

                    #just append the label
                    if self.labels_subset == "all":
                        tmp.append(scp_code)

                    #appends the label correspondent diagnostic class
                    elif self.labels_subset == "superclass":
                        tmp.append(self.scp_statements.loc[scp_code].diagnostic_class)

                    #Appends the correspondent subclass
                    elif self.labels_subset == "subclass":
                        tmp.append(self.scp_statements.loc[scp_code].diagnostic_subclass)

                    else:
                        raise Exception("Invalid subclass set. Choose 'all', 'superclass' or 'subclass'.")
                
                #all other sets of labels dont have either subclass or superclass
                else:
                    tmp.append(scp_code)
        
        #There is no matching class for this example, it will have 0 value at all classes
        if tmp == []:
            return tmp
        else:                     
            #returns a vector and removes duplicates
            return list(set(tmp))
        

        # FALTA ARRANJAR FORMA DE PASSAR PRA ONE HOT ENCODING - primeiro fazer colunas com todas as classes, depois


    #get the dataframe with class and example columns
    def process_dataset(self):
        
        # loads the codes as dictionaries (transforms strings that are like dictionaries into real dictionaries)
        self.database.scp_codes = self.database.scp_codes.apply(lambda x: ast.literal_eval(x))

        #Merge all classes into a column, into a single vector
        self.database ['classes'] = self.database.scp_codes.apply(self.aggregate_diagnostic)

        #Cut the database so we get only the columns we want
        self.dataset = self.database[["patient_id","classes","filename_hr", "strat_fold"]].copy()
        
        #Transforms the column "classes" into multiple columns, one for each class
        self.one_hot_encoding_for_classes()    

    def one_hot_encoding_for_classes(self):
        '''
            Splits the classes from a list into multiple columns and attach them to the self.dataset dataframe
        '''
        #Get the dataframe, the same size as our dataset, filled with 0 and with columns equal to the number of different classes
        df_one_hot = pd.DataFrame(0, index = self.dataset.index, columns = self.target_labels, dtype = int)
        #for each sample of our dataset
        for i, row in self.dataset.iterrows():
            #For each class this ecg is positive to
            
            for class_label in row["classes"]:
                #if the class corresponds to one of the columns
                if class_label in df_one_hot.columns:
                    #assing value 1 to that class
                    df_one_hot.loc[i, class_label] = 1

        self.dataset = pd.concat([self.dataset, df_one_hot], axis=1)

    #Split into train, validation and test set
    def get_train_val_test_set(self):
        
        #training set
        train_set = self.dataset[(self.dataset.strat_fold < 9)] #change this after testing#
        val_set = self.dataset[self.dataset.strat_fold == 9]
        test_set = self.dataset[self.dataset.strat_fold == 10]
        
        
        return train_set,val_set, test_set
    
    def get_full_dataset(self):
        return self.dataset
    
    def get_dataset_targets(self):
        return self.target_labels
    
    def get_dataset_class_count(self, mode = "print"):

        count_classes = (self.dataset == 1).sum()
        count_classes = count_classes.sort_values(ascending=False)

        if mode == "print":
            for col, count in count_classes.items():
                print(f"{col}: {count}")
        elif mode == "return":
            return count_classes
        
        else:
            raise Exception("Invalid mode! Use 'print' or 'return'.")
        

############################################################################## 
#receives an ECG lead, applies preprocessing to it and returns the section of signal selected
def lead_pre_processing(lead, start_point, pp_type = "default", norm= "none", signal_length = 2048):

    '''
        Default transformation:
    '''
    if pp_type == "default":
        #Remove line noise with low pass filter at 50 hz
        lead = butter_lowpass_filter(lead, cutoff, fs, order)

        #Remove baseline wander
        baseline = bwr.calc_baseline(lead)

        clean_lead = torch.sub(torch.tensor(lead.copy()), torch.tensor(baseline))
        
        #Pick just a section of the signal
        clean_lead = pick_signal_section(clean_lead, start_point, signal_length)

    elif pp_type == "bandpass":
        clean_lead = sosfilt(band_pass_filter, lead) #butterworth bandpass 1Hz-45Hz
        clean_lead = torch.tensor(clean_lead)
        clean_lead = pick_signal_section(clean_lead, start_point, signal_length)

    #we will just return the lead as it is
    elif pp_type == "none":
        clean_lead =  pick_signal_section(lead, start_point, signal_length)

    else:
        raise ValueError("The preprocessing method chosen is not implemented!")
    
    #Apply normalization

    if norm == "none":
        #The lead is kept the same
        norm_lead = clean_lead

    #Apply minmax normalization
    elif norm == "minmax":
        
        lead_min = torch.min(clean_lead)
        lead_max = torch.max(clean_lead)

        #Min-Max formula
        norm_lead = (clean_lead - lead_min)/(lead_max-lead_min + 1e-8) #sum 1e-8 to avoid 0/0

    #Apply z-score normalization method
    #signal = (signal-mean)/standard deviation
    elif norm == "zscore":
        lead_mean = torch.mean(clean_lead)
        lead_std = torch.std(clean_lead)

        norm_lead = (clean_lead - lead_mean) / (lead_std + 1e-8) #sum 1e-8 to avoid 0/0

    #robust scaling
    #Apply Robust Scaling (Median & IQR)
    elif norm == "rscal":
        lead_median = torch.median(clean_lead)

        #orders from the smallest to largest value
        #After gets the 25th and 75th percentile
        q1 = torch.kthvalue(clean_lead,int(0.25 * clean_lead.shape[0]))[0] #25th percentile
        q3 = torch.kthvalue(clean_lead,int(0.75 * clean_lead.shape[0]))[0] #75th percentile

        iqr = q3 -q1
        norm_lead = (clean_lead - lead_median) / (iqr + 1e-8)

    #Log scaling
    elif norm == "logscal":
        norm_lead = torch.log1p(torch.abs(clean_lead))

    #L2 normalization
    elif norm == "l2":
        norm_lead = clean_lead / (torch.norm(clean_lead, p=2) + 1e-8)

    else:
        raise Exception("Invalid normalization chosen! Choose 'none', 'minmax' or 'zscore'.")
    

    return norm_lead

#As the signal has a length of 5000, we can choose the length we want to analyse 
#It receives one lead at a time
def pick_signal_section(lead, start_point, signal_length):
    
    return lead[start_point:start_point + signal_length]

#Extract each lead from the xml file and returns a tensor 12-lead ECG with all the transformations required
def get_ecg(input_path = "", test_flag = False, data_aug_flag = True, preprocess = "none", normalization = "none", ecg_len = 2048, max_len = 5000, leads_labels = ['I'], dataset_name = "ptbxl"):
    '''
        Receives 1 ECG (with all leads) at a time and returns the ECG clean, preprocessed and with data augmentation operations applied
        
        input_path:  path to ECG example
        test_flag: if we are using a test example or a training example (test must use always the same starting point)
        data_aug_flag: True if we want data augmentation, false otherwise
        preprocess: which technique we want to apply
        ecg_len: Length of signal we will use in our experiments
        max_len: maximum len of the signal that we can use
        leads_labels: all the labels the ecg dataframe might have. They should be specified so that when iterating through the signal we can get each lead individualy

    '''
    #all datasets are loaded the same way
    if dataset_name == "ptbxl" or dataset_name == "hsm" or dataset_name == "cpsc18":
        #that is the way to load this type of data (default of ptb-xl dataset)
        
        record = wfdb.rdsamp(input_path)

        #Get the dataframe with columns leads labels and values of that ecg lead.
        df = pd.DataFrame(record[0],columns = record[1]["sig_name"])

    else:
        raise Exception("Dataset not implemented! Try 'hsm', 'cpsc18' or 'ptbxl'.")
    
    #keep test set coherent between different runs
    if test_flag == False:
       
       # Get the first point of the signal we will use
       start_point = random.randint(0,max_len-ecg_len) #values possible for the ecg starting point

    #Will try to start at the middle of the signal
    else:
        #If we are using all the signal, the starting point must be
        if ecg_len == max_len:
            start_point = 0

        #We will use the center signal points
        else:
            start_point = int((max_len - ecg_len) / 2)

    #Loop over all leads
    for i in range(len(leads_labels)):
        
        #Get the lead with the corresponding identifier. Vectorized pandas for faster computation
        lead = torch.tensor(df[leads_labels[i]].values, dtype = torch.float32)

        #If padding is needed - add zeros at the end of the signal
        if len(lead) < max_len:
            lead = torch.cat([lead, torch.zeros(max_len - len(lead))])
            
        #If the signal is too big, cut it by the max length
        if len(lead) > max_len:
            lead = lead[:max_len]

        #perform preprocessing of a single lead
        lead = lead_pre_processing(lead, start_point, pp_type = preprocess,norm = normalization, signal_length = ecg_len)

        #Add to the final tensor containing 12 leads
        if i == 0:
            #Add sufficient dimensions for all the leads
            all_leads = torch.unsqueeze(lead, dim = 0)
            
        else:
            #leads_with_noise = torch.cat((leads_with_noise, torch.unsqueeze(lead, dim = 0)), 0)
            all_leads = torch.cat((all_leads, torch.unsqueeze(lead, dim = 0)), 0)
      
    #Perform data augmentation if we want
    if data_aug_flag:
        #Choose the data_augmentation method to perform
        #There are 6 possible data augmentations
        data_aug_type = random.randint(0,6)

        #Perform the selected data augmentation to all the leads
        augmented_leads = dtaug.choose_and_perform_dtaug(all_leads, data_aug_type) 

    #The augmented leads will be just the leads
    else:
        augmented_leads = all_leads
    
    return augmented_leads

## Function to produce all the metrics we want
def get_metrics(all_preds, all_targets, all_probs, class_labels, mode = 'print'):
    '''
        all_preds: network predictions rounded to 1 or 0
        all_targets: real labels
        all_probs: Probabilities of the network predictions
        class_labels: Column tags for labels
        mode: print or return
    '''

    #for binary classification we should do the macro average metrics.

    metrics = {}
    #EXACT accuracy - each target set must match exactly the correspondent predictions
    metrics['acc_exact_match'] = accuracy_score(all_targets, all_preds)

    #micro looks at each label instance
    #From all Positives, how many are really TP
    metrics['precision_overall'] = precision_score(all_targets, all_preds, average = 'macro', zero_division = 0)

    #Percentage of TP caught
    #Same as sensitivity
    metrics['recall_overall'] = recall_score(all_targets, all_preds, average = 'macro', zero_division = 0)

    #F1 score overall instances
    metrics['f1_score_overall'] = f1_score(all_targets, all_preds, average = 'macro', zero_division = 0)

    #AUC receives the probabilities
    #This function builds the ROC curve and coputes the  AUC
    metrics['auc'] = roc_auc_score(all_targets, all_probs, average = 'macro')

    #This is kind of an AUC of a precision recall curve.
    metrics['ap_macro'] = average_precision_score(all_targets, all_probs, average="macro")

    # Compute accuracy per label (column-wise)
    #We can use this to calculate a metric per label
    acc_per_label = []
    precision_per_label = []
    for i in range(len(class_labels)):
        #Get binary pairs of this label
        class_targ = all_targets[:, i]
        class_pred = all_preds[:, i]
        #Get metrics for this class
        acc = accuracy_score(class_targ, class_pred)
        prec =  precision_score(class_targ, class_pred, zero_division = 0)

        #Save the metrics
        acc_per_label.append(acc)
        precision_per_label.append(prec)

    metrics['acc_per_label'] = np.round(acc_per_label, 3)
    metrics['precision_per_label'] = np.round(precision_per_label, 3)

    #Performs more metrics to track specific content
    preds_compressed = all_preds.flatten()  
    targets_compressed = all_targets.flatten() 

    metrics['acc_overall'] = accuracy_score(targets_compressed, preds_compressed)

    #Percentage of TN to all real negatives
    metrics['specificity_overall'] = recall_score(targets_compressed, preds_compressed, pos_label=0, zero_division = 0)

    #Imbalanced does not support multilabel
    #Balance between specificity and sensitivity - the same as recall
    metrics['gmean_overall'] = np.sqrt(metrics['recall_overall'] * metrics['specificity_overall'])




    #If we want to print and return
    if mode == 'print':
        
        print('Validation Metrics:')
        print('Exact match acc: %.3f' % metrics['acc_exact_match'])
        print('Acc overall: %.3f'% metrics['acc_overall'])
        print('Precision: %.3f'% metrics['precision_overall'])
        print('Recall: %.3f'% metrics['recall_overall'])
        print('F1: %.3f'% metrics['f1_score_overall'])
        print('Specificity: %.3f'% metrics['specificity_overall'])
        print('G-Mean: %.3f'% metrics['gmean_overall'])
        print('AP-macro: %.3f'% metrics['ap_macro'])
        print('AUC: %.3f'% metrics['auc'])
        print(class_labels)
        print('Acc: ', metrics['acc_per_label'])
        print('Prec: ', metrics['precision_per_label'])

        return metrics
    
    #just return the metrics, without showing them to the user
    else:
        return metrics


def choose_model(model_type, submodel, n_classes, config):

    model = None

    #Select the model we want to work with

    #AlexNet
    if model_type == "alexnet":
        print("AlexNet chosen")
        model = AlexNet(config['n_leads'], n_classes, config['ecg_len'], config['dropout'])

    #VGG family
    elif model_type == "vgg":
        print("VGG chosen")
        if submodel == "11":
            model = vgg11(config['n_leads'], n_classes, config['ecg_len'], config['dropout'])

        elif submodel == "11_bn":
            model = vgg11_bn(config['n_leads'], n_classes, config['ecg_len'], config['dropout'])

        elif submodel == "16":
            model = vgg16(config['n_leads'], n_classes, config['ecg_len'], config['dropout'])

        elif submodel == "16_bn":
            model = vgg16_bn(config['n_leads'], n_classes, config['ecg_len'], config['dropout'])

        else:
            raise Exception("Invalid submodel!! Not yet implemented, try the implemented ones.")
    
    #Resnet family
    elif model_type == "resnet":
        print("ResNet chosen")
        if submodel == "18":
            model = resnet18(config['n_leads'], n_classes, config['ecg_len'])

        elif submodel == "34":
            model = resnet34(config['n_leads'], n_classes, config['ecg_len'])

        elif submodel == "50":
            model = resnet50(config['n_leads'], n_classes, config['ecg_len'])

        elif submodel == "101":
            model = resnet101(config['n_leads'], n_classes, config['ecg_len'])

        elif submodel == "152":
            model = resnet152(config['n_leads'], n_classes, config['ecg_len'])

        else:
            raise Exception("Invalid submodel!! Not yet implemented, try the implemented ones.")
        
    #HSM-net model
    elif model_type == "attresnet":
        print("1D-Attention ResNet chosen")
        model = AttResNet(config['n_leads'], n_classes, config['ecg_len'], resnet_type = config['resnet_type'], num_heads=8, reduction = config['reduction'])

    elif model_type == "rnn":
        print("RNN chosen")

        model = ECG_RNN(config['n_leads'], 
                        config['rnn_hidden_size'], 
                        config['rnn_layers'], 
                        num_classes = n_classes, 
                        rnn_type = config['submodel'], 
                        dropout = config['rnn_dropout'],
                        bidirectional = config['bidirectional'])
        
    elif model_type == "crnn":
        print("CRNN (ResNet + RNN) chosen")

        model = CRNN(   config['n_leads'], n_classes, config['ecg_len'],
                        resnet_type = config['resnet_type'],
                        hidden_size = config['rnn_hidden_size'], 
                        num_layers = config['rnn_layers'], 
                        rnn_type = config['submodel'], 
                        dropout = config['rnn_dropout'],
                        bidirectional = config['bidirectional'])
        
    elif model_type == "transformer_encoder":
        print('ECG_Transformer Chosen')
        model = ECG_Transformer_Encoder(num_classes = n_classes, 
                                        input_dim = config['n_leads'], 
                                        max_len = config['max_ecg_len'],  
                                        embed_dim = config['embed_dim'], 
                                        num_heads = config['num_heads'], 
                                        num_layers=config['enc_num_layers'], 
                                        enc_dropout = config['enc_dropout'])
        
    elif model_type == "restransformer":
        print('Residual Transformer chosen')
        model = ResTransformer( config['n_leads'], n_classes, config['ecg_len'], 
                                resnet_type = config['resnet_type'], 
                                num_heads = config['num_heads'], 
                                num_layers=config['enc_num_layers'], 
                                enc_dropout = config['enc_dropout'])

    #The chosen model is not implemented
    else:
        raise Exception("Invalid model type!! Not yet implemented, try the implemented ones.")
    
    return model


#Init network weights
def init_weights(m):
    """Custom weight initialization for convolutional, linear, and batch norm layers."""
    if isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  

        if m.bias is not None:
            init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)  # Gamma = 1
        init.constant_(m.bias, 0)    # Beta = 0

    elif isinstance(m, nn.MultiheadAttention):
        # Initialize input projection weight
        init.xavier_uniform_(m.in_proj_weight)
        if m.in_proj_bias is not None:
            m.in_proj_bias.data.fill_(0.0)

        # Initialize output projection weight
        init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            m.out_proj.bias.data.fill_(0.0)

    # Initialize LSTM/GRU/RNN Layers
    elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # Hidden-hidden weights (recurrent weights)
                init.orthogonal_(param)  # Helps with gradient flow
            elif 'bias' in name:  # Biases
                param.data.fill_(0)
                
                # For LSTMs: Set forget gate bias to 1 to prevent vanishing gradients
                if isinstance(m, nn.LSTM):
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)


##################################
## Dataset classes
###################################
# Dataset class for the trainloader
class ptbxl_Dataset(Dataset):
    
    def __init__(self,
                 root_dir, df_dataset,  
                 test = False, 
                 data_aug = False, 
                 norm = "none",
                 pre_process = "none",
                 signal_len = 2048,
                 max_signal_len = 5000,
                 lead_labels = ["I"],
                 dataset_name = "ptbxl" ):
        
      
        #All variables needed to be kept for this class
        self.root_dir = root_dir        #dataset root dir
        self.test = test                #if we are testing or training
        self.data_aug = data_aug        #if we want to perform data augmentation
        self.norm = norm                #if we want to apply normalization
        self.pre_process = pre_process
        self.signal_len = signal_len
        self.max_signal_len = max_signal_len
        self.leads_labels = lead_labels #the labels/tags for each lead of our ECG. 
                                        #only the labels given will be saved
        self.dataset_name = dataset_name


        #Filters just the class columns one hot encoded
        self.labels = df_dataset.iloc[:, 4:]
        #Get the filenames for all dataset samples
        self.file_name = df_dataset['filename_hr']
        self.ecg_path = ""

        
    #Get the len of the datasett
    def __len__(self):
        return len(self.labels)
    
    #Gets one item at a time from the dataframe
    def __getitem__(self,idx):
        
        #Using iloc to access by position and not by index
        self.ecg_path = os.path.join(self.root_dir, str(self.file_name.iloc[idx]))

        label_tensor = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float32)
       
        #Costly in time
        ecg_tensor = get_ecg(input_path = self.ecg_path,
                                    test_flag = self.test, 
                                    data_aug_flag = self.data_aug,
                                    preprocess = self.pre_process,
                                    normalization = self.norm,
                                    ecg_len = self.signal_len,
                                    max_len = self.max_signal_len,
                                    leads_labels = self.leads_labels, 
                                    dataset_name = self.dataset_name                                    
                                    )
        
        #Plot the ecgs
        #plot_ecg(ecg_tensor, leads_to_plot=self.leads_labels, tags = self.leads_labels, sample_rate = 500)
        #sys.exit()
        return (ecg_tensor, label_tensor) 
    
    def get_name(self):
        return self.ecg_path
    

class hsm_Dataset(Dataset):
    
    def __init__(self,
                 dataset_path, database_file_path,  
                 test = False, 
                 data_aug = False, 
                 norm = "none",
                 pre_process = "none",
                 signal_len = 2048,
                 max_signal_len = 5000,
                 lead_labels = ["I"],
                 dataset_name = "ptbxl" ):
        
      
        #All variables needed to be kept for this class
        self.dataset_path = dataset_path       #dataset root dir
        self.test = test                #if we are testing or training
        self.data_aug = data_aug        #if we want to perform data augmentation
        self.norm = norm                #if we want to apply normalization
        self.pre_process = pre_process
        self.signal_len = signal_len
        self.max_signal_len = max_signal_len
        self.leads_labels = lead_labels #the labels/tags for each lead of our ECG. 
                                        #only the labels given will be saved
        self.dataset_name = dataset_name

        database = pd.read_csv(database_file_path, delimiter=';', names = ['filename', 'labels'])

        #column 1 has the labels for each sample
        
        self.labels = database[database.columns[1]]

        #column has the name of file for each sample
        self.file_name = database[database.columns[0]]
        self.ecg_path = ""

        
    #Get the len of the datasett
    def __len__(self):
        return len(self.labels)
    
    #Gets one item at a time from the dataframe
    def __getitem__(self,idx):
        
        #Using iloc to access by position and not by index
        #the filename ends with .xml, so we have to remove that part
        self.ecg_path = os.path.join(self.dataset_path, str(self.file_name.iloc[idx][:-4]))

        label_tensor = torch.tensor(self.labels.iloc[idx], dtype=torch.float32).unsqueeze(0)
    
        #Costly in time
        ecg_tensor = get_ecg(input_path = self.ecg_path,
                                    test_flag = self.test, 
                                    data_aug_flag = self.data_aug,
                                    preprocess = self.pre_process,
                                    normalization = self.norm,
                                    ecg_len = self.signal_len,
                                    max_len = self.max_signal_len,
                                    leads_labels = self.leads_labels, 
                                    dataset_name = self.dataset_name                                    
                                    )
        
        #Plot the ecgs
        #plot_ecg(ecg_tensor, leads_to_plot=self.leads_labels, tags = self.leads_labels, sample_rate = 500)
        
        return (ecg_tensor, label_tensor) 
    
    def get_name(self):
        return self.ecg_path
    
    def get_dataset_targets(self):
        return ["Binary Classification: 1-Has PE, 0-No PE"]
    
#Dataset class for the cpsc dataset
class CPSC18_Dataset(Dataset):
    
    def __init__(self,
                 path_to_data, df_dataset,  
                 test = False, 
                 data_aug = False, 
                 norm = "none",
                 pre_process = "none",
                 signal_len = 2048,
                 max_signal_len = 5000,
                 lead_labels = ["I"],
                 dataset_name = "ptbxl" ):
        
      
        #All variables needed to be kept for this class
        self.dataset_path = path_to_data       #dataset root dir
        self.test = test                #if we are testing or training
        self.data_aug = data_aug        #if we want to perform data augmentation
        self.norm = norm                #if we want to apply normalization
        self.pre_process = pre_process
        self.signal_len = signal_len
        self.max_signal_len = max_signal_len
        self.leads_labels = lead_labels #the labels/tags for each lead of our ECG. 
                                        #only the labels given will be saved
        self.dataset_name = dataset_name
        
        #Get all the labels
        self.labels = df_dataset.iloc[:, 1:-1]

        #column has the name of file for each sample
        #First column has the name of the files
        self.file_name = df_dataset.iloc[:, 0]
        self.ecg_path = ""

        
    #Get the len of the datasett
    def __len__(self):
        return len(self.labels)
    
    #Gets one item at a time from the dataframe
    def __getitem__(self,idx):
        
        #Using iloc to access by position and not by index
        #the filename ends with .xml, so we have to remove that part
        self.ecg_path = os.path.join(self.dataset_path, str(self.file_name.iloc[idx]))

        label_tensor = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float32)
    
        #Costly in time
        ecg_tensor = get_ecg(input_path = self.ecg_path,
                                    test_flag = self.test, 
                                    data_aug_flag = self.data_aug,
                                    preprocess = self.pre_process,
                                    normalization = self.norm,
                                    ecg_len = self.signal_len,
                                    max_len = self.max_signal_len,
                                    leads_labels = self.leads_labels, 
                                    dataset_name = self.dataset_name                                    
                                    )
        
        #Plot the ecgs
        #plot_ecg(ecg_tensor, leads_to_plot=self.leads_labels, tags = self.leads_labels, sample_rate = 500)
        
        return (ecg_tensor, label_tensor) 
    
    def get_name(self):
        return self.ecg_path
    
    def get_dataset_targets(self):
        return ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

def pick_dataset(config):

    if config['dataset'] == "ptbxl":
        
        #Data locations
        path_to_dataset = "/home/guests/jsm/ptblxl_dataset/"

        dataset = prepare_ptbxl_dataset (
                                            path_to_dataset = path_to_dataset,
                                            labels_set = config["set"], 
                                            labels_subset = config["subset"]
        )
        #Process the dataset file
        dataset.process_dataset()

        #getting dataframe for train, validation and test
        df_train, df_val, df_test = dataset.get_train_val_test_set()
    
        #Checking the labels column
        ##########################################################
        #For ptb-xl, these are the labels
        leads_labels = get_ecg_labels(config['dataset'])
        
        #Convert data into dataset class
        print("Creating dataset")
        train_set = ptbxl_Dataset(  path_to_dataset,
                                    df_train,  
                                    test = False, 
                                    data_aug = config["data_aug"], 
                                    norm = config['norm'],
                                    pre_process= config['pre_process'],
                                    signal_len = config['ecg_len'],
                                    max_signal_len = config['max_ecg_len'],
                                    lead_labels = leads_labels,
                                    dataset_name=config['dataset'])
        
        val_set = ptbxl_Dataset(  path_to_dataset,
                                    df_val,  
                                    test = True, 
                                    data_aug = False, 
                                    norm = config['norm'],
                                    pre_process= config['pre_process'],
                                    signal_len = config['ecg_len'],
                                    max_signal_len = config['max_ecg_len'],
                                    lead_labels = leads_labels,
                                    dataset_name=config['dataset'])
        
        test_set = ptbxl_Dataset(  path_to_dataset,
                                    df_test,  
                                    test = True, 
                                    data_aug = False, 
                                    norm = config['norm'],
                                    pre_process= config['pre_process'],
                                    signal_len = config['ecg_len'],
                                    max_signal_len = config['max_ecg_len'],
                                    lead_labels = leads_labels,
                                    dataset_name=config['dataset'])
        
        return dataset, train_set, val_set, test_set
    
    #Load the hsm dataset for pulmonary embolism detection
    elif config['dataset'] == "hsm":
        print('HSM dataset')

        #Data locations
        root_path = "Path_To_Dataset"
        path_to_train = os.path.join(root_path, "all_train_wfdb/")
        path_to_test = os.path.join(root_path, "all_test_wfdb/")
        path_dbtrain_file = os.path.join(root_path,"labels_train_all_wfdb.csv")
        path_dbtest_file = os.path.join(root_path,"labels_test_all_wfdb.csv")

        
        
        leads_labels = get_ecg_labels(config['dataset'])


        print("Creating dataset")
        train_set = hsm_Dataset(path_to_train, path_dbtrain_file,
                                test = False, 
                                data_aug = config["data_aug"], 
                                norm = config['norm'],
                                pre_process= config['pre_process'],
                                signal_len = config['ecg_len'],
                                max_signal_len = config['max_ecg_len'],
                                lead_labels = leads_labels,
                                dataset_name=config['dataset'])
        

        val_set =  hsm_Dataset(path_to_test, path_dbtest_file,
                                test = False, 
                                data_aug = config["data_aug"], 
                                norm = config['norm'],
                                pre_process= config['pre_process'],
                                signal_len = config['ecg_len'],
                                max_signal_len = config['max_ecg_len'],
                                lead_labels = leads_labels,
                                dataset_name=config['dataset'])
        
        test_set =  hsm_Dataset(path_to_test, path_dbtest_file,
                                test = False, 
                                data_aug = config["data_aug"], 
                                norm = config['norm'],
                                pre_process= config['pre_process'],
                                signal_len = config['ecg_len'],
                                max_signal_len = config['max_ecg_len'],
                                lead_labels = leads_labels,
                                dataset_name=config['dataset'])
        
        #this is just needed in the main function to get the number of targets we are evaluating
        dummy_dataset = test_set

        return dummy_dataset, train_set, val_set, test_set


    elif config['dataset'] == "cpsc18":
        print('CPSC18 dataset')
        
        # Root directory for the dataset
        path_to_data = "/home/guests/jsm/cpsc18_dataset/Training_WFDB"
        path_db_file = "/home/guests/jsm/cpsc18_dataset/labels.csv"

        #Get the dataframe for the labels
        df_dataset = pd.read_csv(path_db_file)

        #Fold 8 is used for testing
        df_train = df_dataset[df_dataset.fold != 8]

        #Using the same fold for val and testing for debugging purposes
        df_val = df_dataset[df_dataset.fold == 8]
        df_test = df_dataset[df_dataset.fold == 8]

        # Labels
        leads_labels = get_ecg_labels(config['dataset'])


        print("Creating dataset")
        train_set = CPSC18_Dataset(path_to_data, df_train,
                                test = False, 
                                data_aug = config["data_aug"], 
                                norm = config['norm'],
                                pre_process= config['pre_process'],
                                signal_len = config['ecg_len'],
                                max_signal_len = config['max_ecg_len'],
                                lead_labels = leads_labels,
                                dataset_name=config['dataset'])
        

        val_set =  CPSC18_Dataset(path_to_data, df_val,
                                test = False, 
                                data_aug = config["data_aug"], 
                                norm = config['norm'],
                                pre_process= config['pre_process'],
                                signal_len = config['ecg_len'],
                                max_signal_len = config['max_ecg_len'],
                                lead_labels = leads_labels,
                                dataset_name=config['dataset'])
        
        test_set =  CPSC18_Dataset(path_to_data, df_test,
                                test = False, 
                                data_aug = config["data_aug"], 
                                norm = config['norm'],
                                pre_process= config['pre_process'],
                                signal_len = config['ecg_len'],
                                max_signal_len = config['max_ecg_len'],
                                lead_labels = leads_labels,
                                dataset_name=config['dataset'])
        
        #this is just needed in the main function to get the number of targets we are evaluating
        dummy_dataset = test_set

        return dummy_dataset, train_set, val_set, test_set


    else:
        raise Exception("Dataset not implemented! Try 'ptbxl', 'cpsc18' or 'hsm'.")
    
        
