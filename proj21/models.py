import torch
import torch.nn as nn
from collections import Counter

class NeuralDictionary(nn.Module):
     # Compares a query againts all keys a produces a confidence/probability value for each key, the confidence/probability is multiplied by the value and summed up.
    
     # The model could be speed up with similarity search, or by learning just the Top highest probability/confidence keys/values(or both).
     # We could track which key-value pairs have been learned(with a list of counters or the Counter object) and use that to tell how surprised the network is to see a particular query.
     #   That could be very useful in Reinforcement Learning(as curiosity value) or Classification to detect which class has not been learned.
     #   In Reinforcemenet learning the count of 0 would represent the highest/maximum curiosity value. That would represent a state(or location) that has not been visited. An agent guided by curiosity would try to visit and learn that state.
     #   If the key would represent a Class the count of 0 would suggest that that particular Class has not been learned. 
     #   So byt tracking the count of used(or top confidence) key-value pairs while learning we would learn the uncertainty(or curiosity) values.
     #   Key-value pairs that have not been learned while Trainig the model, that is their attention/confidence value was 0.

    def __init__(self):
        super(Net, self).__init__()
        # 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(500, 100, dtype=torch.double))
        
        # 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(500, 4, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = torch.matmul(self.keys, query)
        attention = torch.softmax(attention, 0)
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        #out = torch.sigmoid(out)
        
        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        #print(self.meta.most_common(10))
      
        return out
    
