# practicing loading data and creating a model with the animals dataset

import pandas as pd

animals = pd.read_csv('deep_learning/zoo.csv')

features = animals.iloc[:, 1:-1]
X = features.to_numpy()

target = animals.iloc[:, -1]
y = target.to_numpy()

import torch
from torch.utils.data import TensorDataset

dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
# float may not be necessary in all scenarios! 

sample = dataset[0]
input_sample, label_sample = sample
print('input sample: ', input_sample)
print('label_sample: ', label_sample)

from torch.utils.data import DataLoader

batch_size = 2 # how many samples we take from the dataset per iter
shuffle = True

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 2),
    nn.Linear(2, 1)
)

output = model(features)
# basic example of a model implementation with a dataset

for batch_inputs, batch_labels in dataloader:
    print('batch inputs', batch_inputs)
    print('batch labels', batch_labels)

hair_ex = animals['hair'] # get specific columns of data

