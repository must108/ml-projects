import torch 
import torch.nn as nn

# activation functions can also be between linear layers
# limitations of sigmoid:
# bounded between 0 and 1, and can be used anywhere in the network
# gradients of the sigmoid function always approach zero for all vals of x
# saturates the function, this leads to vanishing gradients during backpropagation
# this is also a problem for the softmax activation function

# a NEW activation function: ReLU (rectified linear unit)
# outputs maximum between input and 0
# output is equal to the input for positive inputs
# output is equal to zero for negative inputs
# f(x) = max(x, 0)
# this overcomes the problem of vanishing gradients

relu - nn.ReLU()
y = relu(x)
y.backward()
gradient = x.grad
print(gradient)

# leaku reLU is a variation
# behaves similarly with positive inputs
# however, with negative inputs, the input is multiplied by a 
# small coefficient (default at 0.01 with pytorch)
# non null negative gradients!

leaky_relu = nn.LeakyReLU(negative_slope=0.05)




