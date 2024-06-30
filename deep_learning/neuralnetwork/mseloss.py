import numpy as np
import torch
import torch.nn as nn

def mean_squared_loss(prediction, target):
    return np.mean((prediction-target)**2)

criterion = nn.MSELoss()
loss = criterion(prediction, target)