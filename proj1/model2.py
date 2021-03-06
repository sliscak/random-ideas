import torch
from torch import nn

class Net(nn.Module):
    # input is 64 x 64 x 3 image
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64*64*3, 1000)
        self.fc2 = nn.Linear(1000, 64*64*3)

    def forward(self, x):
        x = torch.Tensor(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape((64,64,3))
        return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 64 * 64 input
#         self.fc1 = nn.Linear(64*64, 2000)
#         self.list = nn.Parameter(torch.randn(1000, 1))
#         # self.conv1 = nn.Conv2d(1, 6, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = torch.Tensor(x)
#         x = torch.flatten(x, 0)
#         image = torch.Tensor(x)
#         x = self.fc1(x)
#         x = x.unsqueeze(1)
#         print(x.shape)
#         # x = x * torch.randint(0,2, size=(2000, 1))
#         x = x * self.fc1.weight
#         print(x.shape)
#         x = x.sum(0)
#         print(x.shape)
#         x = torch.sigmoid(x)
#         x = x.reshape((64, 64))
#         return x
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         # x = x.view(-1, 16 * 4 * 4)
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 64 * 64 input
#         self.fc1 = nn.Linear(64*64, 2000)
#         self.list = nn.Parameter(torch.randn(1000, 1))
#         # self.conv1 = nn.Conv2d(1, 6, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = torch.Tensor(x)
#         x = torch.flatten(x, 0)
#         image = torch.Tensor(x)
#         x = self.fc1(x)
#         x = x.unsqueeze(1)
#         print(x.shape)
#         # x = x * torch.randint(0,2, size=(2000, 1))
#         x = x * self.fc1.weight
#         print(x.shape)
#         x = x.sum(0)
#         print(x.shape)
#         x = torch.sigmoid(x)
#         x = x.reshape((64, 64))
#         return x
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         # x = x.view(-1, 16 * 4 * 4)
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 64 * 64 input
#         self.fc1 = nn.Linear(64*64, 2000)
#         self.list = nn.Parameter(torch.randn(1000, 1))
#         # self.conv1 = nn.Conv2d(1, 6, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = torch.Tensor(x)
#         x = torch.flatten(x, 0)
#         image = torch.Tensor(x)
#         x = self.fc1(x)
#         x = x.unsqueeze(1)
#         print(x.shape)
#         # x = x * torch.randint(0,2, size=(2000, 1))
#         x = x * self.fc1.weight
#         print(x.shape)
#         x = x.sum(0)
#         print(x.shape)
#         x = torch.sigmoid(x)
#         x = x.reshape((64, 64))
#         return x
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         # x = x.view(-1, 16 * 4 * 4)
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 64 * 64 input
#         self.fc1 = nn.Linear(64*64, 1000)
#         self.list = nn.Parameter(torch.randn(1000, 1))
#         # self.conv1 = nn.Conv2d(1, 6, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = torch.Tensor(x)
#         x = torch.flatten(x, 0)
#         image = torch.Tensor(x)
#         x = self.fc1(x)
#         print(x.shape)
#         a = torch.argmax(x)
#         print(a)
#         x = self.fc1.weight[a]
#         x = torch.sigmoid(x)
#         x = x.reshape((64, 64))
#         return x
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         # x = x.view(-1, 16 * 4 * 4)
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)

# class Net(nn.Module):
#     def __init__(self, shape):
#         super(Net, self).__init__()
#         # 64 * 64 input
#         self.fc1 = nn.Linear(64*64, 1000)
#         self.list = nn.Parameter(torch.randn(1000, 1))
#         # self.conv1 = nn.Conv2d(1, 6, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = torch.Tensor(x)
#         x = torch.flatten(x, 0)
#         image = torch.Tensor(x)
#         x = self.fc1(x)
#         print(x.shape)
#         x = torch.cosine_similarity(x, self.list, dim=0)
#         print(x.shape)
#         i = torch.argmax(x)
#         x = self.list[i]
#         # x = torch.sigmoid(x)
#         # x = x.reshape(64, 64)
#         x = image * x
#         x = torch.sigmoid(x
#                           )
#         x = x.reshape(64, 64)
#         exit()
#         return x
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         # x = x.view(-1, 16 * 4 * 4)
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)


# class Net(nn.Module):
#     def __init__(self, shape):
#         super(Net, self).__init__()
#         # 64 * 64 input
#         self.fc1 = nn.Linear(64*64, 1000)
#         self.list = nn.Parameter(torch.randn(1000, 1))
#         # self.conv1 = nn.Conv2d(1, 6, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = torch.Tensor(x)
#         x = torch.flatten(x, 0)
#         x = self.fc1(x)
#         print(x.shape)
#         x = torch.cosine_similarity(x, self.list, dim=0)
#         print(x.shape)
#         i = torch.argmax(x)
#         x = self.list[i]
#         print(x)
#         exit()
#         # x = torch.sigmoid(x)
#         # x = x.reshape(64, 64)
#         return x
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         # x = x.view(-1, 16 * 4 * 4)
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)