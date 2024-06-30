import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

animals = pd.read_csv('deep_learning/zoo.csv')

features = animals.iloc[:, 1:-1]
X = features.to_numpy()
# column indexing used to use all data except animal name, and animal type

target = animals.iloc[:, -1]
y = target.to_numpy()
# to get only our target data (in this case, animal type)

dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
# torch.tensor is used to convert data to tensors!

# square bracket indexing can be used for accessing individual samples
sample = dataset[0]
input_sample, label_sample = sample
print('input sample: ', input_sample)
print('label sample: ', label_sample)

batch_size = 2
shuffle = True

# creates a dataloader (obviously...)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

for batch_inputs, batch_labels in dataloader:
    print('batch inputs', batch_inputs)
    print('batch labels', batch_labels)
# loop thru dataloader to get batch inputs and batch labels

# batch size represents the number of samples
# used in a singular forward/backward pass in a neural network




