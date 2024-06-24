# two layer neural network

import torch.nn as nn
import torch

input_tensor = torch.tensor(
    [[0.3471, 0.4547, -0.2356]]
)

linear_layer = nn.Linear(in_features=3, out_features=2)
# the mathematical function performed by .Linear() is shown
# y0 = W0 * X + b0
# where W0 is the weight of the linear layer
# and b0 is the bias of the linear layer 

# essentially:
# output = W0 @ input + b0
# where @ is matrix multiplication

output = linear_layer(input_tensor)
print(output)

# this neural network is fully connected, as there are only linear layers

model = nn.Sequential(
    nn.Linear(10, 18),
    nn.Linear(18, 20),
    nn.Linear(20, 5)
)

inp_tensor = torch.tensor([
    [-0.0014, 0.4038, 1.0305, 0.7521, 
     0.7489, -0.3968, 0.0113, -1.3844, 
     0.8705, -0.9743]
])

output_tensor = model(inp_tensor)
print(output_tensor)