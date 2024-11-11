import torch
import torch.nn as nn

# tensor - matrix or array
# tensors carry derivatives under the hood

a = torch.ones(5, 5)  # 5x5 tensor filled with val 1
print(a)

# sum() and mean() in pyt
sum = torch.sum(a, axis=1)
mea = torch.mean(a, axis=1)
print(sum, mea)

# squeeze and unsqueeze
a = torch.ones(5, 1)
print(a)
squeezed = torch.squeeze(a)
print(squeezed)
unsqueezed = torch.unsqueeze(squeezed, dim=1)
print(unsqueezed)


# neural network models
# nn models inherits from nn.module

# nn.Linear nodes do linear regression

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(4, 6)
        self.second_layer = nn.Linear(6, 6)
        self.output_layer = nn.Linear(6, 2)

    def forward(self, x):
        return self.output_layer(self.second_layer(self.first_layer(x)))


model = MyModel()

# random numbers in tensor
example_datapoint = torch.randn(1, 4)

print(model(example_datapoint))

# train the model, and then make predictions
