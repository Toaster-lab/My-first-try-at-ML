'''We use Pytorch for deeplearning.

Definition of deeplearning after Google: "Deep learning is a subset of machine learning 
that uses multi-layered artificial neural networks to analyze complex data 
and automatically discover patterns without manual feature extraction."

We use features(Inputs) and labels(output) to get the relationship between those.

Why we use maschine learning/deeplearning:
-For a complex problem, can you find all the rules? 
-We can use it for baically anything to find a pattern aslong as you can convert it to numbers.

How to install Pytorch:
"pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126" <- With Cuda support, which makes it faster
'''
import torch
import pandas as pd
import numpy as np

##creating tensors

#scalar: -> doesnt have dimentions
scalar = torch.tensor(7)
#print(scalar.ndim) -> means it is just one number, aka like a point
#print(scalar.item()) -> turns it back to a integer

#vector -> means it is like a line which has magnitude and direction
vector = torch.tensor([7,7])
#print(vector.ndim) -> means it is like a line, 1 dimension
#print(vector.shape) -> 2 elements

#MATRIX
MATRIX = torch.tensor([[1,2],
                       [3,4]])
#print(MATRIX.ndim) -> 2, means it is like a square
#print(MATRIX.shape) -> OUT: torch.Size([2, 2]) | this means that the first 'vector' has two elements and the second also 2

#TENSOR
TENSOR = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,8,9],
                        [10,11,12]]])
#print(TENSOR.ndim) -> 3, it is a cube of data
#print(TENSOR.shape) -> OUT: torch.Size([1, 3, 3]) | means one 3x3 tensor

'''
Why random tensors?
Random tensors are important becaue the way many neural networks learn is that
they start with tensors full of random numbers and then adjust those random numbers
to better represent the data.
Start with random numbers -> look at data -> update random numbers -> update random numbers
'''
#Random tensors
random_tensor = torch.rand(3, 4) # 3 rows and 4 elements each

#How pictures get made to tensors
image_tensor = torch.rand(size=(3, 224, 224)) #3 colour channels, height, width or random picture

#Create a tensor of all zeros or ones
zero = torch.zeros(size=(3,4))
ones = torch.ones(size=(3,4))
#print(ones.dtype) -> default data type is torch.flaot32

#Creating a range of tensors and tensors-like
range = torch.arange(start=0,end=10, step=1)
#print(range) -> works like the range in python

tensors_like = torch.zeros_like(input=range)
#print(tensors_like) -> adapt the shape of a tensor without keeping its data

#Tensor datatypes -> one of the big 3 errors with PyTorch & deep learning
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,           #what dataype
                               device='cuda',          #"cpu" by default, "cuda" for faster computing
                               requires_grad=False)  #PyTorch trackes gradient with this tensors operations

''' Big 3 Errors
1. Tensor not right datatype - to get it: tensor.dtype
2. Tensor not right shape    - to get it: tensor.shape/torch.size()
3. Tensors not on the right device - to get it: tensor.device  '''

float_16_tensor = float_32_tensor.type(torch.float16) #torch.half
#print(float_16_tensor)


#Getting information/attribute from tensors
#print(float_32_tensor.device)

#Manipulating tensors (tensor operations)
'''Tensor operations: Addition, Subratction, 
Multiplication, Division, Matrix Multiplication
There is two main ways of performing multiplication in nn and dl
1. Element-wise mulitplication -> a*b
2. Matrix multiplication (dot product) -> torch.matmul(a,b)
'''
tensor = torch.tensor([[1,2,3],
                      [4,5,6]])
#print(tensor*tensor)
#print(torch.matmul(tensor,tensor)) -> short: torch.mm()
#The two main rules are in matmul: 1. The inner dimensions must match. 2. The resulting matrix has the shape of the outer dimensions

'''How to manipulate shape of tensor: We use transpose: tensor.T 
This changes the shape f. e. ([3,2]) to ([2,3]) -> rows become coloumns
'''
#print(tensor.T) -> OUT: tensor([[1, 4],[2, 5],[3, 6]])
#tensor.type(torch.float32) <- changes dtype

#Tensor aggregation
#minimun: torch.min(tensor), tensor.min()
#Maximum: torch.max(tensor), tensor.max()
#Mean:    torch.mean(tensor), tensor.mean() <- to find the mean you need a float or a complex
#Sum:     torch.sum(tensor), tensor.sum()

 #Finding the positional min and max tensors
 #tensor.argmin()
 #tensor.argmax()

 #Reshaping, stacking, squeezing and unsqueezing tensors
'''
 -Reshaping - reshapes an input tensor to a defined shape
 -View - Return a view of a input of certain shape but
   keep the samem eory as the original tensor
- Stacking - combine multible tensors on top of each otehr (vstack)
   or side by side
- Squeeze - removes all '1' dimensions from a tensor
- Unsqueeze - add '1' dimension to a target tensor
- Permute - Return a view of the input with dimensions permuted (swapped)
   in a certain way
 '''

#creating tensor
x = torch.arange(start=1,end=10, step=1)
x = x.type(torch.float32)
#print(x, x.shape)
#Reshape
#x_reshaped = x.reshape(x, y) -> x*y always has to be the original size 
x_reshaped = x.reshape(1,9)

#Change the view
#z = x.view(1, 9) -> shares the same memory as x, if you change z, x changes 
#z[:, 0] = 5

#Stacking tensors on top of each other
#x_stacked = torch.stack([x,x,x,x,x], dim=0) -> dim=1 makes torch.Size([5, 9]) to torch.Size([9, 5])

# torch.squeeze() - removes all single dimensions from a target tensor
#print(x_reshaped.shape) -> ([1,9]) it has a single dimensions
x_squeezed = x_reshaped.squeeze() # -> torch.Size([9])
print(x_squeezed.unsqueeze(dim=0))

# torch.permute - rearranges the dimensions of a target tensor in a specified order
x_image = torch.rand(size=(224,224,3))
 #-> permute the original tensor to rearrange the axis (or dim) order
x_permuted = x_image.permute(2, 0, 1) #shifts axis 0->1, 1->2, 2->0 
# they share the same memory
print(x_permuted.shape, x_image.shape)

