import torch
from torch import nn

# instead of using a dense layer i used trainable parameters(weights) and cosine similarity
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


net = Net(3, 10)
input = torch.tensor([0,1,0])
output = net(input)
print(output)