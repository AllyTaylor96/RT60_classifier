"""
Main training function for the rt60 classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import pickle
from utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from models.resnet import resnet_model
from pathlib import Path
import argparse
from training_functions import *

# set up parser
parser = argparse.ArgumentParser(description='Training rt60 classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('train_folder', help='Destination of training files')
args = parser.parse_args()

"""
Load in the dataset 
"""
data_direc = args.train_folder
dataset = DatasetFolder(data_direc, pt_loader, ('.pt'))

"""
Pickle class indexes
"""
classifier_classes = dataset.find_classes(data_direc)[0]
pickle.dump(classifier_classes, open("results/classes", "wb"))

"""
Split the dataset into test and train
"""
train_size = int(0.9 * len(dataset))
valid_size = len(dataset) - train_size
print('Training size: {} \nValidation size: {}'.format(train_size, valid_size))
train_dataset, valid_dataset = torch.utils.data.random_split(dataset,
                                                            [train_size, valid_size])

"""
Load data into torch
"""
params = {'batch_size': 32, 'shuffle': True} # can amend these
print('Batch size: {}'.format(params['batch_size']))
training_generator = torch.utils.data.DataLoader(train_dataset, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset, **params)

"""
Send to device
"""
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

"""
Set up hyperparameters, variables for training and import model
"""
learning_rate = 0.0001
num_epochs = 25
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=learning_rate)
resnet_model = resnet_model.to(device)
count = 0
train_losses = []
valid_losses = []
train_history = {}
accuracy_history = {}
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                       patience = 5)
"""
Main training loop below
"""
print('Beginning Training...')
train(resnet_model, loss_fn, training_generator, validation_generator,
      num_epochs, optimizer, train_losses, valid_losses, train_history,
      accuracy_history, scheduler)
path = '2s_rt60_classifier.pt'
torch.save(resnet_model, path)

"""
Save results and plot the history
"""
with open('results/2s_results_bs{}_lr{}_epochs{}.txt'.format(params['batch_size'], learning_rate, num_epochs), 'w') as f:
    f.write('Training Loss per epoch: \n')
    for key in train_history.keys():
        f.write("'{}':'{}'\n".format(key, train_history[key]))
    f.write('Training Accuracy per epoch: \n')
    for key in accuracy_history.keys():
        f.write("'{}':'{}'\n".format(key, accuracy_history[key]))

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(list(train_history.values()), lw=3)
ax.set_title('Training Loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(list(accuracy_history.values()), lw=3)
ax.set_title('Training Accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.savefig('results/2s_training_graph_bs{}_lr{}_epochs{}.png'.format(params['batch_size'], learning_rate, num_epochs))
