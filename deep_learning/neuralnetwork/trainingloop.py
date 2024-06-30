import torch
import torch.nn as nn
from torch import TensorDataset, DataLoader
from torch.optim import optim
import numpy as np

# creates dataset and dataloader
dataset = TensorDataset(torch.tensor(features).float(), torch.tensor(target).float())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# create the model
model = nn.Sequential(nn.Linear(4, 2),
                      nn.Linear(2, 1))

# creates loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# loop thru the dataset multiple times
# every loop thru a dataset is called an "epoch"
for epoch in range(num_epochs):
    for data in dataloader:
        # set gradients to zero
        optimizer.zero_grad()

        # get feature and target from the dataloader
        feature, target = data
        
        # run a forward pass
        pred = model(feature)

        # compute loss and gradients
        loss = criterion(pred, target)
        loss.backward()

        # update the parameters
        optimizer.step()
# shows the results
show_results(model, dataloader)


# extra code

y_hat = np.array(10)
y = np.array(1)

# calculate MSELoss with numpy and y-hat and y values
mse_numpy = np.mean((y_hat-y)**2)

# create an MSELoss function
criterion = nn.MSELoss()

# calculates MSELoss with a created loss function
mse_pytorch = criterion(torch.tensor(y_hat).float(), torch.tensor(y).float())
