import torch
import torch.nn as nn

input_data = torch.tensor(
    [
        [-0.4421, 1.5207, 2.0607, -0.3647, 0.4691, 0.0946],
        [-0.9155, -0.0475, -1.3645, 0.6336, -1.9520, -0.3398],
        [0.7406, 1.6763, -0.8511, 0.2432, 0.1123, -0.0633],
        [-1.6630, -0.0718, -0.1285, 0.5396, -0.0288, -0.8622],
        [-0.7413, 1.7920, -0.0883, -0.6685, 0.4745, -0.4245]
    ]
)

model = nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

output = model(input_data)
print(output)
# binary classification^

n_classes = 3
model = nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, n_classes),
    nn.Softmax(dim=-1)
)

output = model(input_data)
print(output.shape)
print(output)
# multi-class classification^

model = nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, 1)
)

output = model(input_data)
print(output)
# regression with forward pass
