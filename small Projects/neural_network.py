'''import numpy as np

input = np.array([1, 2, 3])
weights = np.array([3.1, 4.5, 6.1])
biases = np.array([5, 1.2, 6])

output = np.dot(input, weights) + biases
print(output)

target_value = 30
loss = np.zeros(3)

for i in range(3):
    loss[i] = target_value - output[i]

print(loss) '''
#How to write this in torch?
import torch

inputs = torch.tensor([1,2,3],
                      dtype=torch.float32)
weights =torch.tensor([3.1, 4.5, 6.1],
                      dtype=torch.float32)
biases = torch.tensor([5, 1.2, 6],
                      dtype=torch.float32)

output = torch.dot(inputs, weights) + biases
#no need for torch.zeros because torch doesnt 
#need a existing tensor to put the loss in there it does it automatically
target_value = 30
loss = target_value - output
print(loss)