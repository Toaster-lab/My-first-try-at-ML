#Second half of the one hour course
import torch
import numpy as np
X = torch.randn(size=(10, 1))
## torch.nn:

# torch.nn.linear
D_in = 1
D_out = 1

linear_layer = torch.nn.Linear(in_features=D_in, out_features=D_out)

#print(linear_layer.weight, linear_layer.bias)

y_hat_nn = linear_layer(X)
#print(y_hat_nn)

# ReLu
relu = torch.nn.ReLU()
sample_data = torch.tensor([-2.0,-.5,0,0.5,2.0])
activation_relu = relu(sample_data)
#GELU
gelu = torch.nn.GELU()
activation_gelu = gelu(sample_data)

# Softmax
softmax = torch.nn.Softmax(dim=-1)
logits = torch.tensor([[1.0, 3.0, 0.5, 1.5],
                       [-1.0, 2.0, 1.0, 0.0]])
activation_softmax = softmax(logits)
print(activation_softmax)


# NN.Embedding
vocab_size = 10
embedding_dim = 3

embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

# Input: A sentence where each word is an ID, 1 5 0 8
input_ids = torch.tensor([[1, 5, 0, 8]])
word_vectors = embedding_layer(input_ids)
#This creates the vectors for the words
print(word_vectors)

# NN.LayerNorm: values dont explode or vanish
# Normalizes features to stabilize training

# Our word vectors have a feature dimension of 3
norm_layer = torch.nn.LayerNorm(normalized_shape=3)

input_features = torch.tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])

normalized_features = norm_layer(input_features)
print(normalized_features)