
# dataset must be split into training data, validation, and testing
# (you know this!)

# training data is used to adjust the model's parameters (weights and biases)

# validation data is used for hyperparameter tuning.

# testing data is used once to calculate final metrics

# focus heavily on evaluating loss in training/validation
# as well as accuracy in training/validation



# calculate training loss by summing up the los for 
# each iteration of the training set dataloader.
# calculate the mean training loss at the end of each epoch

import pandas as pd 
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

animals = pd.read_csv('deep_learning/zoo.csv')

features = animals.iloc[:, 1:-1]
X = features.to_numpy()

target = animals.iloc[:, -1]
y = target.to_numpy()

dataset = TensorDataset(
    torch.tensor(X).float(), 
    torch.tensor(y).float())

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


