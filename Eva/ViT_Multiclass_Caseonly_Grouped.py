from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision as tv
from torchvision import datasets, models, transforms
from sklearn.metrics import roc_auc_score

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import time
import os
import copy
import requests
import io
import csv
import random
from timm.data import create_dataset, create_loader
from timm.scheduler import StepLRScheduler


plt.ion()   # interactive mode

import timm 
from tqdm import tqdm

# Check if CUDA (GPU support) is available
is_cuda_available = torch.cuda.is_available()
print("Is CUDA (GPU) available:", is_cuda_available)

# If CUDA is available, print the GPU name(s)
if is_cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPU(s) available: {gpu_count}")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")
    
# config 
input_size = 3, 224, 224
img_size = 224
num_classes = 8
batch_size = 128

interpolation = 'bicubic'
DEFAULT_CROP_PCT = 1

train_dir = '../Dataset/images/train_caseonly_grouped'
val_dir = '../Dataset/images/validation_caseonly_grouped'

class_map = {
    'Fluid_overload': 0,
    'Infection': 1,
    'Mass_Like_Lesions': 2,
    'Parenchymal_Disease': 3,
    'Atelectasis': 4,
    'Cardiomegaly': 5,
    'Pneumothorax': 6,
    'Pleural_Thickening': 7
    }

# create the train and eval datasets
train_dataset = create_dataset(name='', root=train_dir, split='train', is_training=True, batch_size=batch_size, class_map = class_map)
val_dataset = create_dataset(name='', root=val_dir, split='validation', is_training=False, batch_size=batch_size, class_map = class_map)
train_len, val_len = len(train_dataset), len(val_dataset)
print('Training set size: ' + str(train_len))
print('Validation set size: ' + str(val_len))

# resize images to fit the input of pretrained model
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    #transforms.CenterCrop((224*3, 224*3)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset.transform = transform
val_dataset.transform = transform

# create data loaders 
loader_train = create_loader(
        train_dataset,
        input_size=input_size,
        batch_size=batch_size,
        is_training=True,
        interpolation=interpolation,
        num_workers=8,
        pin_memory=True)

loader_val = create_loader(
        val_dataset,
        input_size=input_size,
        batch_size=batch_size,
        is_training=False,
        interpolation=interpolation,
        num_workers=8,
        pin_memory=True,
        crop_pct=DEFAULT_CROP_PCT)

# check if labels are loaded as defined
train_dataset.reader.class_to_idx

# check how many images for each class. confirm if this number is correct to make sure images are loaded properly
class_images_num = dict(zip(class_map.values(),[0]*15))
for i in range(len(train_dataset.reader)):
    _, class_idx = train_dataset.reader[i]
    class_images_num[class_idx] += 1

print(class_images_num)

# define model
model_name = 'vit_base_patch16_224.orig_in21k'

model = timm.create_model(model_name, pretrained=True, num_classes=num_classes) #, img_size=img_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) # if print 'cuda' then GPU is used
model.to(device)

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Move model to GPU
    model.cuda()

    # Move data tensors to GPU
    #data_tensor = data_tensor.cuda()

    # Check the device of model parameters
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device: {param.device}")

    # Check the device of data tensor
    print(f"loader_train is on device: {loader_train.device}")
    print(f"loader_val is on device: {loader_val.device}")

else:
    print("CUDA (GPU) is not available.")
    
    
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

# weight_decay
skip = {}
if hasattr(model, 'no_weight_decay'):
    skip = model.no_weight_decay()
parameters = add_weight_decay(model, 0.0001, skip)
weight_decay = 0.

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(parameters, momentum=0.9, nesterov=True, lr=0.01, weight_decay=weight_decay)

# setup learning rate schedule and starting epoch
lr_scheduler = StepLRScheduler(optimizer, decay_t=30, decay_rate=0.1,
               warmup_lr_init=0.0001, warmup_t=3, noise_range_t=None, noise_pct=0.67,
               noise_std=1., noise_seed=42)

def eval_fn(model, eval_data):
    model.eval()

    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch, (images, labels) in enumerate(eval_data):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
    accuracy = total_correct / total_samples
    
    return accuracy

# define number of epoch and create empty list to store 
num_epochs = 30
losses = [[]]
accus_train = [[]]
accus_val = []
learning_rates = []

from datetime import datetime
current_datetime = datetime.now()
date_time = str(current_datetime)[:-7].replace('-','').replace(':','').replace(' ','_')

model_save_path = f'model_result/model_pth/MODEL_FINETUNE_{model_name}_{date_time}.pth'
print(model_save_path)

def output_log_writer(s, end='\r'):
    with open(f'model_result/log/output_{model_name}_{date_time}.txt', 'a') as output_file:
        output_file.write(s+'\n')
        print(s,end,flush=True)
        
for epoch in range(num_epochs):
    output_log_writer(f'-------------------------------[Epoch {epoch+1}]---------------------------------')
    output_log_writer(f'[Epoch {epoch+1}] Training...', end='')
    
    for batch, (images, labels) in enumerate(loader_train):
        print('=', end='')
        
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses[-1].append(loss.item()) # all losses for this epoch
        
        with torch.no_grad():
            accus_train[-1].append(torch.sum(torch.max(outputs, dim=1)[1] == labels)) # all train accuracy for this epoch

    
    print('\r')
    output_log_writer(f'[Epoch {epoch+1}] Computing Train Measurement...')
    # save all batches loss
    with open(f'model_result/measurement/batch_loss_{model_name}_{date_time}.csv', 'a') as b_loss_file:
        writer = csv.writer(b_loss_file)
        writer.writerow([epoch] + losses[-1])
    
    # compute total loss [train]
    losses[-1] = sum(losses[-1]) 
    # save epoch loss
    with open(f'model_result/measurement/epoch_loss_{model_name}_{date_time}.csv', 'a') as e_loss_file:
        writer = csv.writer(e_loss_file)
        writer.writerow([epoch, losses[-1]])
    losses.append([])
    
    # compute accuracy and auc [train]
    accus_train[-1] = sum(accus_train[-1]) / train_len # accuracy
    with open(f'model_result/measurement/acc_train_{model_name}_{date_time}.csv', 'a') as acc_train_file:
        writer = csv.writer(acc_train_file)
        writer.writerow([epoch, float(accus_train[-1])])
    accus_train.append([])

    # step LR for next epoch
    output_log_writer(f'[Epoch {epoch+1}] Update Learning Rate...')
    lr = optimizer.param_groups[0]['lr']
    learning_rates.append(lr)
    lr_scheduler.step(epoch + 1)
    # save learning rate for this epoch
    with open(f'model_result/measurement/learning_rate_{model_name}_{date_time}.csv', 'a') as lr_file:
        writer = csv.writer(lr_file)
        writer.writerow([epoch, learning_rates[-1]])
    
    # compute accuracy and auc [validation]
    output_log_writer(f'[Epoch {epoch+1}] Computing Validation Measurement...')
    accuracy_val = eval_fn(model, loader_val)
    accus_val.append(accuracy_val)
    with open(f'model_result/measurement/eval_val_{model_name}_{date_time}.csv', 'a') as eval_val_file:
        writer = csv.writer(eval_val_file)
        writer.writerow([epoch, accus_val[-1]])
    
    model.train() # slowest line
    
    # print evaludation
    output_log_writer(f'[Epoch {epoch+1}] loss={losses[-2]:.2e} train accu={accus_train[-2]:.2%} validation accu={accus_val[-1]:.2%}')
        
    # save model checkpoint
    ckpt_save_path = f'model_result/model_checkpoint/MODEL_CKPT_{epoch}_{model_name}_{date_time}.pt'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses[-2],
            }, ckpt_save_path)
    output_log_writer(f'[Epoch {epoch+1}] Model checkpoint is saved.')
    
        
torch.save(model, model_save_path)
output_log_writer('\n\nFinal model saved. Training Finished.')



