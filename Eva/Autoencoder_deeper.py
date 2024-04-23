from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import torchvision as tv
from torchvision import datasets, models, transforms

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

import time
import os
import copy
import requests
import io
import csv

plt.ion()   # interactive mode

import timm 
from tqdm import tqdm


# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3 * 224 * 224, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * 224 * 224),
            nn.Sigmoid()
        )
 
 
    def forward(self, x):
        batchsz = x.size(0)
        # flatten
        x = x.view(batchsz, -1)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(batchsz, 3, 224, 224)
        
        return x  
    

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
    
from timm.data import create_dataset, create_loader

# config 
input_size = 3, 224, 224
img_size = 224
num_classes = 15
batch_size = 128

hidden_dim = 128
latent_dim = 32
learning_rate = 0.001     # 学习率
num_epochs = 50     # 迭代次数

interpolation = 'bicubic'
DEFAULT_CROP_PCT = 1

train_dir = '../Dataset/images/train'
val_dir = '../Dataset/images/validation'

class_map = {
        'Normal': 0,
        'Atelectasis': 1,
        'Cardiomegaly': 2,
        'Effusion': 3,
        'Infiltration': 4,
        'Mass': 5,
        'Nodule': 6,
        'Pneumonia': 7,
        'Pneumothorax': 8,
        'Consolidation': 9,
        'Edema': 10,
        'Emphysema': 11,
        'Fibrosis': 12,
        'Pleural_Thickening': 13,
        'Hernia': 14,
        }

# create the train and eval datasets
train_dataset = create_dataset(name='', root=train_dir, split='train', is_training=True, batch_size=batch_size, class_map = class_map)
val_dataset = create_dataset(name='', root=val_dir, split='validation', is_training=False, batch_size=batch_size, class_map = class_map)
train_len, val_len = len(train_dataset), len(val_dataset)
print('Training set size: ' + str(train_len))
print('Validation set size: ' + str(val_len))

# resize images to fit the input of pretrained model
transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset.transform = transform
val_dataset.transform = transform

# create data loaders 
train_loader = create_loader(
        train_dataset,
        input_size=input_size,
        batch_size=batch_size,
        is_training=True,
        interpolation=interpolation,
        num_workers=8,
        pin_memory=True)

val_loader = create_loader(
        val_dataset,
        input_size=input_size,
        batch_size=batch_size,
        is_training=False,
        interpolation=interpolation,
        num_workers=8,
        pin_memory=True,
        crop_pct=DEFAULT_CROP_PCT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) # if print 'cuda' then GPU is used

autoencoder = Autoencoder(latent_dim,hidden_dim).to(device)

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Move model to GPU
    autoencoder.cuda()

    # Move data tensors to GPU
    #data_tensor = data_tensor.cuda()

    # Check the device of model parameters
    for name, param in autoencoder.named_parameters():
        print(f"Parameter {name} is on device: {param.device}")

    # Check the device of data tensor
    print(f"train_loader is on device: {train_loader.device}")
    print(f"val_loader is on device: {val_loader.device}")

else:
    print("CUDA (GPU) is not available.")

    
# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

from datetime import datetime
current_datetime = datetime.now()
date_time = str(current_datetime)[:-7].replace('-','').replace(':','').replace(' ','_')
model_name = 'Autoencoder_2Layers'

def output_log_writer(s, end='\r'):
    with open(f'model_result/log/output_{model_name}_{date_time}.txt', 'a') as output_file:
        output_file.write(s+'\n')
        print(s,end,flush=True)
        
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    output_log_writer(f'-------------------------------[Epoch {epoch+1}]---------------------------------')
    output_log_writer(f'[Epoch {epoch+1}] Training...', end='')
    autoencoder.train()
    running_loss = 0.0
    for images, _ in train_loader:
        print('=', end='')
        images = images.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(images)
        
        #output_log_writer(f'[Epoch {epoch+1}] Computing Train Loss...')
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    loss_train = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {loss_train}")
    
    # save model checkpoint
    ckpt_save_path = f'model_result/model_checkpoint/MODEL_CKPT_{epoch}_{model_name}_{date_time}.pt'
    torch.save({
            'epoch': epoch,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_train,
            }, ckpt_save_path)
    output_log_writer(f'[Epoch {epoch+1}] Model checkpoint is saved.')

# Save model
torch.save(autoencoder, f'model_result/model_pth/MODEL_{model_name}_{date_time}.pth')
print('Autoencoder training complete.')
