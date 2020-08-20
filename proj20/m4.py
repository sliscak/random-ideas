# Network based on similarity search and retrieval
import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Sigmoid()
        )
        self.memory = nn.ParameterList([nn.Parameter(torch.rand((10,))) for i in range(3)])

    def forward(self, feature):
        print(self.memory[0])
        feature = self.encoder(feature)
        sim_list = torch.cosine_similarity(feature, self.memory[0], 0).unsqueeze(0)
        print(f'x: {feature}')
        print(f'c: {self.memory}')
        print(f'Memory Tensors: {[str(t) for t in self.memory]}')
        for i in range(2):
            sim_list = torch.cat((sim_list, torch.cosine_similarity(feature, self.memory[i + 1], 0).unsqueeze(0)))
        print(f'x: {sim_list}')
        output = torch.softmax(sim_list, dim=0)
        return output

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