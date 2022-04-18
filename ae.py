from turtle import forward
import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # activation functions
import torch.optim as optim  # optimizer
from torch.autograd import Variable # add gradients to tensors
from torch.nn import Parameter # model parameter functionality

import torchvision
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
import random

# tells PyTorch to use an NVIDIA GPU, if one is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ##################################
# import data
# ##################################
# download the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# separate into data and labels
# training data

train_data = mnist_trainset.data.to(dtype=torch.float32)
train_data = train_data.reshape(-1, 784)
train_data /= 255.0
train_labels = mnist_trainset.targets.to(dtype=torch.long)

print("train data shape: {}".format(train_data.size()))
print("train label shape: {}".format(train_labels.size()))

# testing data
test_data = mnist_testset.data.to(dtype=torch.float32)[:2000]
test_data = test_data.reshape(-1, 784)
test_data /= 255.0
test_labels = mnist_testset.targets.to(dtype=torch.long)[:2000]

print("test data shape: {}".format(test_data.size()))
print("test label shape: {}".format(test_labels.size()))

# def min_max_norm(data): 
#     data = (data - data.min()) / (data.max() - data.min())

# min_max_norm(test_data)
# min_max_norm(train_data)


# load into torch datasets
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

# ##################################
# set hyperparameters
# ##################################
# Parameters
learning_rate = 0.001 # Ha ha! This means it will learn really quickly, right?
num_epochs = 100
batch_size = 64

# Network Parameters
n_hidden_1 = 128  # 1st layer number of neurons
n_hidden_2 = 16
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

#regularization
lp = 1
l = 0

loss = 0

max_accuracies = [0.0, 0.0, 0]

# ##################################
# defining the model 
# ##################################

# method 1: define a python class, which inherits the rudimentary functionality of a neural network from nn.Module

class AE(torch.nn.Module):
    def __init__(self, input_dim, code_dim):
        super().__init__()

        self.nonlin = torch.nn.Tanh()
        self.nonlin2 = torch.nn.Tanh()

        self.layer1 = torch.nn.Linear(input_dim, n_hidden_1)
        self.layer2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = torch.nn.Linear(n_hidden_2, code_dim)
        self.layer4 = torch.nn.Linear(code_dim, n_hidden_2)
        self.layer5 = torch.nn.Linear(n_hidden_2, n_hidden_1)
        self.layer6 = torch.nn.Linear(n_hidden_1, input_dim)

    def encode(self, x):
      x.to(device)
      z = self.nonlin(self.layer1(x))
      z = self.nonlin(self.layer2(z))
      return self.nonlin(self.layer3(z)).to(device)
    
    def decode(self, x):
      x.to(device)
      z = self.nonlin(self.layer4(x))
      z = self.nonlin(self.layer5(z))
      return self.nonlin2(self.layer6(z)).to(device)

    def forward(self, x):
      # print(x)
      # x = self.nonlin2(x)
      # print(x)
      code = self.encode(x)
      # print(code)
      return self.decode(code).to(device)


# ##################################
# creata dataloader
# ##################################
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

# These might be handy
MSELoss = nn.MSELoss()
BCELoss = nn.BCELoss()


# ##################################
# main training function
# ##################################

def train():

    model = AE(num_input, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    outputs = []

    print("Starting training")
    # iterate over epochs
    for ep in range(num_epochs):
        model.train()
        # print("start epoch")
        for batch_indx, batch in enumerate(trainloader):
            # print("start image", image.shape)
            # unpack batch
            data, labels = batch

            optimizer.zero_grad()

            image = data.to(device)

            reconstructed = model(image)

            loss = MSELoss(reconstructed, image)

            loss.backward()
            optimizer.step()
        
        outputs.append((num_epochs, image, reconstructed))

        # print loss every 100 epochs
        if ep % 10 == 0:
            print(f"\tcurr loss: {loss} at epoch: {ep}")

    
    print(f"\tcurr loss: {loss} at epoch: {ep}")
    return outputs, model


if __name__ == '__main__':
    outputs, model = train()

    model.eval()
    '''
    plot 2 autoencoder values, with color corresponding to the label.
    '''
    random_ids = np.random.randint(len(test_dataset), size=100000)
    plot_set = test_dataset[random_ids][0].to(device)
    plot_labels = test_dataset[random_ids][1]
    # print(plot_set[0])

    code = model.encode(plot_set)
    code = code.detach().cpu()

    print(code[:5])
    colors = ['r', 'g', 'b', 'y', 'orange', 'purple', 'pink', 'black', 'brown', 'slategrey']
    plt.scatter([x[0] for x in code], [x[1] for x in code], color=np.array(colors)[plot_labels])
    plt.show()