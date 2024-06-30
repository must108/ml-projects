import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss # most used loss function!

# loss function tells us the efficiency of the model during training
# takes in y-hat and y
# outputs float!

# goal is to minimize loss!

# loss = F(y, y-hat)

# one hot encoding can be used to convert integers to tensors
# one hot encoding of 0 when n_classes = 3:
# np.array([1, 0, 0])

print(F.one_hot(torch.tensor(0), num_classes=3))
print(F.one_hot(torch.tensor(1), num_classes=3))
print(F.one_hot(torch.tensor(2), num_classes=3))

scores = torch.tensor([[-0.1211, 0.1059]])
one_hot_target = torch.tensor([1, 0])

criterion = CrossEntropyLoss()
print(criterion(scores.double(), one_hot_target.double()))
