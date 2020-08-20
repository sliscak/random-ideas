# unlimited classes for neural networks
import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Sigmoid()
        )
        # self.classes = list([torch.random((10,)) for i in range(3)])
        self.classes = nn.ParameterList([torch.rand((10,)) for i in range(3)])
        # self.classes = torch.Tensor([torch.rand((10,)) for i in range(3)])
        # self.classes = torch.rand((3, 10))
        #self.classes = [torch.rand(10,) for i in range(3)]

    def forward(self, x):
        print(self.classes[0])
        x = self.block(x)
        # x = torch.Tensor([torch.cosine_similarity(x, self.classes[i], dim=0) for i in range(3)])
        # x2 = torch.cosine_similarity(x, self.classes[0], dim=0)
        # for i in range(2)
        x2 = torch.cosine_similarity(x, self.classes[0], 0).unsqueeze(0)
        # x2 = torch.nn.functional.mse_loss(x, self.classes[0], 0).unsqueeze(0)
        # x3 = torch.cosine_similarity(x, self.classes[1], 0).unsqueeze(0)
        print(f'x: {x}')
        print(f'c: {self.classes}')
        for i in range(2):
            x2 = torch.cat((x2, torch.cosine_similarity(x,self.classes[i+1], 0).unsqueeze(0)))
            # x2 = torch.cat((x2, torch.nn.functional.mse_loss(x, self.classes[i], 0).unsqueeze(0)), 0)
        x = x2
        # x = 1/x2
        # print(f'f: {torch.nn.functional.mse_loss(x, self.classes[i], 0)}')
        print(f'x: {x}')
        x = torch.softmax(x, dim=0)
        return x

x_train = [0.1, 0.5, 0.9]
y_train = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

net = Net()
# o = net(torch.Tensor([x_train[0]]))
# print(o)

optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001, )
criterion = torch.nn.MSELoss()
from time import sleep
while True:
    for i in range(len(x_train)):
        o = net(torch.Tensor([x_train[i]]))
        loss = criterion(o, torch.Tensor([y_train[i]]))
        print(f'O: {o.detach()}')
        print(f'Y: {y_train[i]}')
        print(f'Loss: {loss.detach()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sleep(0.1)