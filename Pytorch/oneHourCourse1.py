#First half of the one hour course
import torch
import numpy as np

# Pattern 1: Direct creation from data
data = [[1.,2.,3.], [4.,5.,6.]]
my_tensor = torch.tensor(data)

# Pattern 2: Creatig from a desired shape
shape = (2, 3)
ones = torch.ones(shape)
zeros = torch.zeros(shape)
random = torch.randn(shape)

# Pattern 3: Creation by mimicking another tensor
template = torch.tensor([[1,2],[3,4]])
rand_like = torch.rand_like(template, dtype=torch.float32)

# The 3 critical attributes
tensor = torch.randn(2, 3)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)
#Note: we use float because for gradients we must use float numbers 
#so weights and biases must be float type


# Data vs Parameter: requires_grad: True (data.requires_grad)
w = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True) #live recording now

#Building the graph
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(4.0, requires_grad=True)
#Goal: compute z = x*y, where y = a + b
y = a + b
z = x * w
#print(z, z.grad_fn) 
#<MulBackward0 object at 0x000002E7D5B1D2D0> it comes from operation

# Operations / Verbs from PyTorch
# @ - matrix multiplication

# Reduction operations
# dim: which direction to collapse
# 2D Tensor: dim=0 -> vertically, dim=1 -> horizontally 

# Basic Indexing
# basic indexing [:,1]
# .argmax -> where are the largest values OUTPUT: (1,3)
# .gather() -> (data, what dim, what indices to select)

# The forward pass
# Calculates prediction and builds computation graph

# Training data: y = 2x + 3
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_true = 2 * X + 3

# Hyperparameters
learning_rate, epochs = 0.01, 1000

# Initialize parameters
W = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Training loop
for epoch in range(epochs):
    # Forward pass and loss
    y_hat = X @ W + b
    loss = torch.mean((y_hat - y_true)**2)

    # Backward pass
    loss.backward()

    # Update parameters
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad
