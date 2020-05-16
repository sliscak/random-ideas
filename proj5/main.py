from proj5.model import AdaptNet
from time import sleep
import torch

adaptnet = AdaptNet()
optimizer = torch.optim.Adam(params=adaptnet.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

c = 0
while True:
    c += 1
    x_train = torch.rand((1,))
    output = adaptnet(x_train)
    loss = criterion(output, x_train.detach())
    print(f'Input: {x_train.detach()}\nOutput: {output.detach()}\nLoss: {loss.detach()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('*'*10)
    sleep(1)
    # print(optimizer.param_groups)
    # exit()
    # if c >= 2:
    #     print(len(list(dynet.parameters())))
    #     exit()