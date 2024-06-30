import torch
import torch.nn as nn

input_tensor = torch.tensor(
    [[4.3, 6.1, 2.3]]
)

probabilities = nn.Softmax(dim=-1)
# dim = -1 indicates that softmax is applied to the
# last dimension of the input tensor
output_tensor = probabilities(input_tensor)

print(output_tensor)

# softmax is used for multi-class classification
# different from sigmoid as it does more than binary classification
# can be the last layer in nn.Sequential()