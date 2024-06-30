import torch
import torch.nn as nn 
from torch.nn import CrossEntropyLoss

model = nn.Sequential(nn.Linear(16, 8),
                      nn.Linear(8, 4),
                      nn.Linear(4, 2))
prediction = model(sample)

# beginning backpropagation
criterion = CrossEntropyLoss()
loss = criterion(prediction, target)
loss.backward()

# to update model parameters
lr = 0.001
weight = model[0].weight
weight_grad = model[0].weight.grad
weight = weight - lr * weight_grad

# update the biases
bias = model[0].bias
bias_grad = model[0].bias.grad
bias = bias - lr * bias_grad

# loss functions in deep learing are non-convex (multiple local minima)
# optimizers take care of gradient descent (iterative model parameter updating)
# most common optimizer is called stochasic gradient descent (SGD)

import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()
