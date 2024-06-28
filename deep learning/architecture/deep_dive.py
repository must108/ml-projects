# linear layers are fully connected!

# each neuron of a layer is connected 
# to each neuron of a previous layer
# (in a neural network...)

# a neuron in a linear layer...

# * computs a linear operation using all
# * the neurons of a previous layer

# contains n+1 learnable parameters

# fully connected neural networks contain three types of layers...
# * the input layer, the hidden layers, and the output layer

# the number of input and output layers are fixed
# the input layer is dependent on the number of features (n_features)
# the output layer is dependent on the number of categories (n_classes)
# see the code below.

import torch.nn as nn 
n_features = 3
n_classes = 2
# ^(non-meaningful vals in this case)
model = nn.Sequential(nn.Linear(n_features, 8),
                      nn.Linear(8, 4),
                      nn.Linear(4, n_classes))

# we can use as many hidden layers as we want
# increasing the number of hidden layers increases the
# number of parameters which increases model capacity (works with more complex datasets)

model = nn.Sequential(nn.Linear(8, 4),
                      nn.Linear(4, 2))

# the first layer as 4 neurons, each with 8+1 params
# 36 parameters
# the second layer as 2 neurons, each with 4+1 params 
# 10 parameters
# total of 46 parameters
# .numel() will return the number of elements in the tensor
# see code below

total = 0
for parameter in model.parameters():
    total += parameter.numel()
print(total)
# this will give us the total number of parameters in a network

