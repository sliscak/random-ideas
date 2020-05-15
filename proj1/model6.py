import torch
from torch import nn

class Net(nn.Module):
    # input is 64 x 64 x 3 image
    def __init__(self):
        super(Net, self).__init__()
        self.p = nn.Parameter(torch.randn(100), requires_grad=True)
        self.p2 = nn.Parameter(torch.randn(32 * 32 * 3), requires_grad=True)
        #
        # # use sigmoid
        self.forget_gate = nn.Linear(100, 100)
        # # use tanh
        self.add_gate = nn.Linear(100, 100)
        #
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc2 = nn.Linear(100, 32*32*3)

        self.ff1 = nn.Linear(100, 100)
        self.ff2 = nn.Linear(100, 1)

    def forward(self, img):
        x = self.fc1(img)
        g = self.ff1(x)
        i = self.ff2(g)
        i = torch.sigmoid(i)
        c = 0
        while(i < 1):
            c += 1
            fg = self.forget_gate(x)
            fg = torch.sigmoid(x)
            ag = self.add_gate(x)
            ag = torch.tanh(x)
            self.p.data = self.p * fg
            self.p.data = self.p + ag
            g = self.ff1(x)
            g = self.ff2(g)
            g = torch.sigmoid(g)
            i = i + g
            print(i)
        print(f'Counter: {c}, i={i}')
        # x = torch.cat([x, p])
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.reshape((32,32,3))
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