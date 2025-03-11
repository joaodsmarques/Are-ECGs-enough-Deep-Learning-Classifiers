# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 21:22:35 2023

@author: Utilizador
"""

import torch
import torch.nn as nn
import sys
import random
import torch.nn.functional as F
from numpy import arange, sin, pi, linspace
from numpy.random import normal
from scipy.signal import square




#Note: random.random() - returns a random value between 0 and 1

# Selects the data augmentation technique and applies it.
#curr lead: lead chosen - chosen: data augmentation chosen technique - 
#random_fact: can be a probability or constant for the ecg

def choose_and_perform_dtaug (all_leads, chosen):
    
    
    #Perform no augmentation
    if chosen == 0 or chosen ==1:
        return all_leads
    
    elif chosen == 2:
        #return all_leads
        return Flip(all_leads)
    
    elif chosen == 3:
        return Random_drop(all_leads)
    
    elif chosen ==4:
        return Lead_drop(all_leads)
    
    elif chosen == 5:
        return Add_sine(all_leads)
    
    elif chosen == 6:
        return Add_square_pulse(all_leads)
        
    
    else:
        print('No augmentation technique was chosen')
        sys.exit()


def Flip(leads):
    return torch.mul(leads, -1)

#Randomly drops (assigns 0) to 10%of each lead sample
def Random_drop(leads):
    
    #probability of dropping 
    p = 0.1
    if len(leads) > 1:
        drop_mask = torch.ones(len(leads[0]))
        #Multiplication prevents dropout from changing values we want to be kept
        drop_mask = torch.mul(F.dropout(drop_mask, p = p), (1-p))
    else:
        print('Single leads not yet supported!')
    
    return torch.mul(leads, drop_mask)

#Randomly drops 20% of the leads
def Lead_drop(leads):
    
    #probability of dropping a lead
    p = 0.2
    for i in range(len(leads)):
        if random.random() < p:
            leads[i] = torch.zeros(len(leads[i]))
    
    return leads
    
#This is useless if we perform z-score norm
def ECG_scale(leads):
    minimum = 0.25
    #make random scale
    magnitude = torch.tensor(random.uniform(minimum, 1))
    
    lead = torch.mul(leads, magnitude)
    
    return leads

#Frequency is random between [0.001, 0.02].
#Common values for `magnitude` are between [0, 1].
def Add_sine(leads):
    
    magnitude = random.uniform(0.25, 1)
    frequency = 0.019 * random.random() + 0.001
    #Length of each lead
    samples = arange(len(leads[0]))
    
    #sin function
    signal =  torch.tensor(magnitude * sin(2 * pi * frequency * samples))
    #mp.plot(samples, signal, 'Sinusoide')
    
    for i in range(len(leads)):
        
        #mp.plot(samples, leads[i], 'before')
        leads[i] = torch.add(leads[i], signal)
        #mp.plot(samples, leads[i], 'after')
        
    return leads

#Adds a square pulse to the ECG
def Add_square_pulse(leads):
    """
    Adds square pulses to the signal with random frequency and amplitude `magnitude`.
    Frequency is random between [0.001, 0.1].
    Common values for `magnitude` are between [0, 0.02]. 
    """
    magnitude = random.uniform(0.005, 0.02)
    frequency = 0.099 * random.random() + 0.001
    
    #Length of each lead
    samples = arange(len(leads[0]))
    
    square_pulse = torch.tensor(magnitude * square(2 * pi * frequency * samples))
    #mp.plot(samples, square_pulse, 'pulse')
    
    for i in range(len(leads)):
        
        #mp.plot(samples, leads[i], 'before')
        leads[i] = torch.add(leads[i], square_pulse)
        #mp.plot(samples, leads[i], 'after')
        
    return leads

