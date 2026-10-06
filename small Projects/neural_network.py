import numpy as np

input = np.array([1, 2, 3])
weights = np.array([3.1, 4.5, 6.1])
biases = np.array([5, 1.2, 6])

output = np.dot(input, weights) + biases
print(output)

target_value = 30
loss = np.zeros(3)

for i in range(3):
    loss[i] = target_value - output[i]

print(loss)