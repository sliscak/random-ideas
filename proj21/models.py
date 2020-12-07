import torch
import torch.nn as nn
from collections import Counter

class NeuralDictionary(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 500 keys each of size 100
        self.keys = nn.Parameter(torch.randn(500, 100, dtype=torch.double))
        
        # 500 values each of size 4
        self.values = nn.Parameter(torch.randn(500, 4, dtype=torch.double))

        # to later see how many times a key has been chosen as the most important
        self.meta = Counter()

    def forward(self, query):
        attention = torch.matmul(self.keys, query)
        attention = torch.softmax(attention, 0)
        out = torch.matmul(attention, self.values)
        out = torch.sigmoid(out)
        
        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        #print(self.meta.most_common(10))
      
        return out
    
