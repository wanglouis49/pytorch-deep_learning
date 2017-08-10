import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
	'''
	Defining a neural network
	'''

	def __init__(self):
		super(Net, self).__init__()
		# 1 input image channel, 6 output channels, 5x5 square convolution kernel
		self.conv1 = nn.Conv2d(1, 6, 5)
		# 6 input image channel, 16 output channels, 5x5 square convolution kernel
		self.conv2 = nn.Conv2d(6, 16, 5)
		# affine operations: y = Wx + b
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		'''
		The forward function is defined here. The backward function (where 
		gradients are computed) is automatically defined using autograd.
		Can use any of the Tensor operations in the forward function.
		'''
		# Max pooling over a (2, 2) window from the first conv layer
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# Max pooling over a (2, 2) window from the second conv layer
		# If the size is a square you can only specify a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		# Stretch the last conv layer
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		'''
		Flattening the tensor
		'''
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


if __name__ == "__main__":
	# Initialize a nn
	net = Net()
	print(net)

	# The learnable parameters of a model are returned by net.parameters()
	params = list(net.parameters())
	print(len(params))
	print(params[0].size())    # conv1's weight

	# The input to the forward is an autograd.Variable, and so is the output
	# Only supports mini-batches
	# Input: nSamples x nChannels x Height x Width
	input = Variable(torch.randn(1, 1, 32, 32))
	out = net(input)
	print(out)

	# Zero the gradient buffers of all parameters
	net.zero_grad()
	# Backprops with random gradients
	out.backward(torch.randn(1, 10))

	# Loss function: takes the (output, target) pair of inputs, and computes
	# a value that estimates how far away the output is from the target
	output = net(input)
	target = Variable(torch.range(1, 10))
	criterion = nn.MSELoss()

	loss = criterion(output, target)
	print(loss)

	# Backprop
	net.zero_grad()

	print('conv1.bias.grad before backward')
	print(net.conv1.bias.grad)

	loss.backward()

	print('conv1.bias.grad after backward')
	print(net.conv1.bias.grad)

	# Update the weights
	optimizer = optim.SGD(net.parameters(), lr=0.01)

	optimizer.zero_grad()
	output = net(input)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()







