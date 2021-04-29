import torch
from torch import nn
from torch.optim import AdamW

class Mem(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.param = nn.Parameter(torch.randn(output_size, input_size), requires_grad=True)

    def forward(self, x):
        x = x.unsqueeze(0)
        o = torch.cosine_similarity(self.param, x)
        return o

class Net(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = Mem(input_size, output_size)

    def forward(self, x):
        x = self.layer(x)
        return x


net = Net(3, 2)
optimizer = AdamW(net.parameters(), lr=0.001)
criterion = nn.MSELoss()


while True:
    x = torch.tensor([0,1,0], dtype=float)
    y = torch.tensor([1,0], dtype=float)
    output = net(x)
    loss = criterion(output, y)
    print(f'Loss: {loss.detach()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
