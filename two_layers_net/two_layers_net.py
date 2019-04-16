import os
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from load_data.load_data import _load_folder


def simple_gradient():
    # print the gradient of 2x^2 + 5x
    x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
    z = 2 * (x * x) + 5 * x
    # run the backpropagation
    z.backward(torch.ones(2, 2))
    print(x.grad)


def load_drive(batch_size):
    dirname = os.path.dirname(__file__)
    p = os.path.expanduser('../data/drive')
    # path = os.path.join(dirname, 'data','drive')
    x_train, masks, manual = _load_folder(p, True)
    x_test, masks = _load_folder(p, False)

    x_test_buckets = map_to_tensor(batch_size, x_test)
    x_train_buckets = map_to_tensor(batch_size, x_train)

    return x_test_buckets, x_train_buckets


def map_to_tensor(batch_size, x_test):
    x_test_buckets = [x_test[x:x + batch_size] for x in range(0, len(x_test), batch_size)]
    random.shuffle(x_test_buckets)
    x_test_buckets = list(map(lambda x: torch.tensor(x), x_test_buckets))
    listofzeros = torch.tensor( [random.randrange(1, 10) for _ in range(10)])
    x_test_buckets = list(map(lambda x: (x, listofzeros), x_test_buckets))
    return x_test_buckets



def create_nn(batch_size=5, learning_rate=0.01, epochs=10,
              log_interval=10):
    # test_loader2, train_loader2 = load_mnist_data(batch_size)
    test_loader, train_loader = load_drive(batch_size)

    data_example = 'a'
    target_example = 'a'
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx == 0 :
            data_example = data
            target_example= target

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x)

    net = Net()
    print(net)

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.NLLLoss()

    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            d1 = data.size()[1]
            d2 = data.size()[2]
            d3 = data.size()[3]
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            # data is a Tensor of size( batch_size=200, 28*28)
            # data = data.view(-1, 28 * 28)
            data = data.view(-1, d1 * d2 * d3)

            optimizer.zero_grad()
            # net out is a Tensor of (batch_size=200,classes=10)
            net_out = net(data)
            # loss is a Tensor of size 1, the loss value
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data))

    # run a test loop
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 28 * 28)
        net_out = net(data)
        # sum up batch loss
        test_loss += criterion(net_out, target).data
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def load_mnist_data(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnsit', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)
    return test_loader, train_loader


if __name__ == "__main__":
    run_opt = 2
    if run_opt == 1:
        simple_gradient()
    elif run_opt == 2:
        create_nn()
