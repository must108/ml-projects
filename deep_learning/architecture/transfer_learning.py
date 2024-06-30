import torch.nn as nn

layer = nn.Linear(64, 128)
print(layer.weight.min(), layer.weight.max())
# intially, you should initialize layer weights to small values,
# like in this case

layer = nn.Linear(64, 128)
nn.init.uniform_(layer.weight)
print(custom_layer.fc.weight.min(), custom_layer.fc.weight.max())
# weight values range from 0-1

# transfer learning - reusing a model trained for a first task
# for a second similar task, to accelerate the training process

# fine-tuning - a type of transfer learning
# we load weights from a previous trained model 
# train the model with a smaller learning rate
# we can also freeze layers in the network!
# a general rule of thumb is to
# freeze earlier layers in a network and fine-tune
# layer closer to the output layer
# see code below!

import torch.nn as nn

model = nn.Sequential(nn.Linear(64, 128),
                      nn.Linear(128, 256))

for name, param in model.named_parameters():
    if name == '0.weight':
        param.requires_grad = False

# code to freeze layers of a model:
# (assume the network has 1 input, 1 hidden, and 1 output layer)

for name, param in model.named_parameters():
    # check the first linear layer
    if name == '0.weight' or name == '0.bias':
        param.requires_grad = False

    # check the second linear layer
    if name == '1.weight' or name == '1.bias':
        param.requires_grad = False


