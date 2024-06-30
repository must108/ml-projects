import torch
import torch.nn as nn

input_tensor = torch.tensor([[6.0]])
sigmoid = nn.Sigmoid()
output = sigmoid(input_tensor)
print(output)

model = nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# sigmoid processes data passed through linear layers
# by making them a value between 0 and 1 (this has been seen before!)
# this allows us to easily determine outcomes of binary features (0 or 1)