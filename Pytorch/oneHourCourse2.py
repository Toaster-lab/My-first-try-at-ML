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

# NN.droput: prevent overfitting -> randomly zeros neurons during training
# This prevent one neuron to have too much impact

dropout_layer = torch.nn.Dropout(p=0.5) #-> We zero out 50 percent of them

# nn.Module -> organize our model / standard structure
# torch.optim -> automate the learning / gradients
 
# THE MODEL BLUEPRINT: 'NN.MODULE'
# Inherit from torch.nn.Module -> __init__: define layers
#                              -> forward: connect layers

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # In the consructor, we DEFINE the layers we will sue
        self.linear_layer = torch.nn.Linear(in_features, out_features)
    def forward(self, x):
        # We connect the layers
        return self.linear_layer(x)
# Instantiate the model
model = LinearRegressionModel(in_features=1, out_features=1)
# torch.optim
import torch.optim as optim

learning_rate = 0.01

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()
loss = 1 # So later no Error thrown :)
# THE THREE-LINE MANTRA
# 1.
optimizer.zero_grad()

# 2.
loss.backward()

# 3.
optimizer.step()


#The final, clean training loop

# Example training data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = 2 * X + 3

epochs = 100

for epoch in range(epochs):
    # Forward pass
    y_hat = model(X)

    # Calculate loss
    loss = loss_fn(y_hat, y_true)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# THE FIVE-STEP LOGIC IS UNIVERSAL
prediction = model(X)             # 1. Forward pass
loss = loss_fn(prediction, y_true) # 2. Calculate error
optimizer.zero_grad()             # 3. Reset gradients
loss.backward()                   # 4. Backpropagation
optimizer.step()                  # 5. Update parameters