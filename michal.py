import torch
#from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torchvision import datasets, transforms

####### ----------------- U-net ----------------- #######
class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()

        # Input Tensor Dimensions = 256x256x3
        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        nn.init.xavier_uniform(self.conv1.weight)  # Xaviers Initialisation
        self.activ_1 = nn.ELU()
        # Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        # Output Tensor Dimensions = 128x128x16


        # Input Tensor Dimensions = 128x128x16
        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        nn.init.xavier_uniform(self.conv2.weight)
        self.activ_2 = nn.ELU()
        # Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        # Output Tensor Dimensions = 64x64x32

        # Input Tensor Dimensions = 64x64x32
        # Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        nn.init.xavier_uniform(self.conv3.weight)
        self.activ_3 = nn.ELU()
        # Output Tensor Dimensions = 64x64x64

        # 32 channel output of pool2 is concatenated

        # https://www.quora.com/How-do-you-calculate-the-output-dimensions-of-a-deconvolution-network-layer
        # Input Tensor Dimensions = 64x64x96
        # De Convolution 1
        self.deconv1 = nn.ConvTranspose2d(in_channels=96, out_channels=32, kernel_size=3, padding=1)  ##
        nn.init.xavier_uniform(self.deconv1.weight)
        self.activ_4 = nn.ELU()
        # UnPooling 1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
        # Output Tensor Dimensions = 128x128x32

        # 16 channel output of pool1 is concatenated

        # Input Tensor Dimensions = 128x128x48
        # De Convolution 2
        self.deconv2 = nn.ConvTranspose2d(in_channels=48, out_channels=16, kernel_size=3, padding=1)
        nn.init.xavier_uniform(self.deconv2.weight)
        self.activ_5 = nn.ELU()
        # UnPooling 2
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2)
        # Output Tensor Dimensions = 256x256x16

        # 3 Channel input is concatenated

        # Input Tensor Dimensions= 256x256x19
        # DeConvolution 3
        self.deconv3 = nn.ConvTranspose2d(in_channels=19, out_channels=1, kernel_size=5, padding=2)
        nn.init.xavier_uniform(self.deconv3.weight)
        self.activ_6 = nn.Sigmoid()
        ##Output Tensor Dimensions = 256x256x1

    def forward(self, x):
        out_1 = x
        out = self.conv1(x)
        out = self.activ_1(out)
        size1 = out.size()
        out, indices1 = self.pool1(out)
        out_2 = out
        out = self.conv2(out)
        out = self.activ_2(out)
        size2 = out.size()
        out, indices2 = self.pool2(out)
        out_3 = out
        out = self.conv3(out)
        out = self.activ_3(out)

        out = torch.cat((out, out_3), dim=1)

        out = self.deconv1(out)
        out = self.activ_4(out)
        out = self.unpool1(out, indices2, size2)

        out = torch.cat((out, out_2), dim=1)

        out = self.deconv2(out)
        out = self.activ_5(out)
        out = self.unpool2(out, indices1, size1)

        out = torch.cat((out, out_1), dim=1)

        out = self.deconv3(out)
        out = self.activ_6(out)
        out = out
        return out

####### ----------------- 2 layers FC net ------------------- #######
# def simple_gradient():
#     # print the gradient of 2x^2 + 5x
#     x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
#     z = 2 * (x * x) + 5 * x
#     # run the backpropagation
#     z.backward(torch.ones(2, 2))
#     print(x.grad)


# def create_nn(batch_size=200, learning_rate=0.01, epochs=10,
#               log_interval=10):
#
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])),
#         batch_size=batch_size, shuffle=True)
#
#     class Net(nn.Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.fc1 = nn.Linear(28 * 28, 200)
#             self.fc2 = nn.Linear(200, 200)
#             self.fc3 = nn.Linear(200, 10)
#
#         def forward(self, x):
#             x = F.relu(self.fc1(x))
#             x = F.relu(self.fc2(x))
#             x = self.fc3(x)
#             return F.log_softmax(x)
#
#     net = Net()
#     print(net)
#
#     # create a stochastic gradient descent optimizer
#     optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
#     # create a loss function
#     criterion = nn.NLLLoss()
#
#     # run the main training loop
#     for epoch in range(epochs):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = Variable(data), Variable(target)
#             # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
#             data = data.view(-1, 28*28)
#             optimizer.zero_grad()
#             net_out = net(data)
#             loss = criterion(net_out, target)
#             loss.backward()
#             optimizer.step()
#             if batch_idx % log_interval == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(data), len(train_loader.dataset),
#                            100. * batch_idx / len(train_loader), loss.data))
#
#     # run a test loop
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         data, target = Variable(data, volatile=True), Variable(target)
#         data = data.view(-1, 28 * 28)
#         net_out = net(data)
#         # sum up batch loss
#         test_loss += criterion(net_out, target).data
#         pred = net_out.data.max(1)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data).sum()
#
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#
#
# if __name__ == "__main__":
#     run_opt = 2
#     if run_opt == 1:
#         simple_gradient()
#     elif run_opt == 2:
#         create_nn()

