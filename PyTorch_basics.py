import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

#========================== Table of Contents ==========================#
# 1. Basic autograd example 1               (Line 21 to 36)
# 2. Basic autograd example 2               (Line 39 to 77)
# 3. Loading data from numpy                (Line 80 to 83)
# 4. Implementing the input pipline         (Line 86 to 113)
# 5. Input pipline for custom dataset       (Line 116 to 138)
# 6. Using pretrained model                 (Line 141 to 155)
# 7. Save and load model                    (Line 158 to 165) 

#======================= Basic autograd example 1 =======================#
# Create tensors.
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 