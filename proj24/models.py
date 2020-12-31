import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, infeatures:int, capacity:int):
        super(Net, self).__init__()
        self.memory = nn.ParameterList([nn.Parameter(torch.rand(infeatures)) for i in range(capacity)])

    def forward(self, x):
        distances = torch.tensor([])
        for i in range(len(self.memory)):
            distance = torch.cosine_similarity(x, self.memory[i], dim=0)
            # print(distance)
            # print(distances)
            distances = torch.cat((distances, torch.unsqueeze(distance, dim=0)), dim=0)
        return distances

class Net2(nn.Module):
    def __init__(self, infeatures:int, capacity:int):
        super(Net2, self).__init__()
        self.memory = nn.ParameterList([nn.Parameter(torch.rand(infeatures)) for i in range(capacity)])

    def forward(self, x):
        distances = torch.tensor([])
        for i in range(len(self.memory)):
            distance = torch.cosine_similarity(x, self.memory[i], dim=0)
            # print(distance)
            # print(distances)
            distances = torch.cat((distances, torch.unsqueeze(distance, dim=0)), dim=0)
        attention = torch.softmax(distances, dim=0)
        return attention

class Net3(nn.Module):
    def __init__(self, infeatures:int, capacity:int):
        super(Net3, self).__init__()
        self.memory = nn.ParameterList([nn.Parameter(torch.rand(infeatures)) for i in range(capacity)])

    def forward(self, x):
        distances = torch.tensor([])
        for i in range(len(self.memory)):
            distance = torch.cosine_similarity(x, self.memory[i], dim=0)
            # print(distance)
            # print(distances)
            distances = torch.cat((distances, torch.unsqueeze(distance, dim=0)), dim=0)
        argmax = torch.argmax(distances, dim=0)
        mask = torch.zeros(distances.shape[0], dtype=torch.double) # could use "pytorch.sparse" here to maximize efficiency
        mask[argmax] = 1
        out = distances * mask
        return out

class Net4(nn.Module):
    def __init__(self, infeatures:int, capacity:int):
        super(Net4, self).__init__()
        self.memory = nn.ParameterList([nn.Parameter(torch.rand(infeatures)) for i in range(capacity)])

    def forward(self, x):
        distances = torch.tensor([])
        for i in range(len(self.memory)):
            distance = torch.cosine_similarity(x, self.memory[i], dim=0)
            # print(distance)
            # print(distances)
            distances = torch.cat((distances, torch.unsqueeze(distance, dim=0)), dim=0)
        argmax = torch.argmax(distances, dim=0)
        out = torch.zeros(distances.shape[0], dtype=torch.double) # could use "pytorch.sparse" here to maximize efficiency
        out[argmax] = distances[argmax]
        return out

class NeuralDictionaryV9(nn.Module):
    # Dictionary where the key is static(nontrainable) and the value is a learnable(trainable) parameter.
    # All keys are saved inside the index.
    # Use the update method to add key-value pairs.
    # Keys and values can be of a different size(shape) but all subsequent keys and values must be of the same size(shape) as the first key and value respectively.
    # Returns the value from the key-value pair, for which the key is most similar to the query.
    # (the algorithm finds the most similar key to a query and then returns the value of the key-value pair which had the most similar key.)

    def __init__(self, in_features: int, out_features: int, capacity: int):
        super(NeuralDictionaryV9, self).__init__()
        self.keys = nn.ParameterList([nn.Parameter(torch.rand(in_features, dtype=torch.double)) for i in range(capacity)])
        self.values = nn.ParameterList([nn.Parameter(torch.rand(out_features, dtype=torch.double)) for i in range(capacity)])

    def forward(self, query):
        distances = torch.tensor([])
        for i in range(len(self.keys)):
          distance = torch.cosine_similarity(query, self.keys[i], dim=0)
          distances = torch.cat((distances, distance.unsqueeze(0)), dim=0)
        argmax = torch.argmax(distances)
        out = self.values[argmax]
        return out
        # q = torch.unsqueeze(query, 0).detach().numpy().astype('float32')
        # distances, ids = self.index.search(q, 1)
        # print(f'IDS: {ids}')
        # id = ids[0]
        # return self.values[id]

    # def update(self, key, value):
        # key = torch.unsqueeze(key, 0)
        # self.index.add(key.detach().numpy().astype('float32'))
        # value = torch.unsqueeze(value, 0)
        # print(value)
        # if self.values is None:
        #     self.values = nn.Parameter(value)
        # else:
        #     self.values = nn.Parameter(torch.cat((self.values, value)))
net = NeuralDictionaryV9(2, 2, 3)
x = torch.tensor([1, 2], dtype=torch.double)
out = net(x.detach())
print(out)
from torch.optim import AdamW
from torch.nn import MSELoss

optimizer = AdamW(net.parameters(), lr=0.01)
criterion = MSELoss()
y = torch.tensor([-1, 2], dtype=torch.double)

for i in range(3):
    out = net(x.detach())
    loss = criterion(y.detach(), out)
    loss.backward()
    optimizer.step()
    print(f"LOSS: {loss.detach()}\tKEYS: {list(net.keys)}\tVALUES: {list(net.values)}")

class NeuralDictionaryV10(nn.Module):
    # Learning by memorizing evidence.
    def __init__(self, in_features: int, out_features: int, capacity: int):
        super(NeuralDictionaryV10, self).__init__()
        self.values = torch.tensor([])
        self.index = faiss.IndexFlatL2(in_features)

    def forward(self, query):
        q = torch.unsqueeze(query, 0).detach().numpy().astype('float32')
        distances, ids = self.index.search(q, 1)
        print(f'IDS: {ids}')
        id = ids[0]
        return self.values[id]

    def update(self, key, value):
        key = torch.unsqueeze(key, 0)
        self.index.add(key.detach().numpy().astype('float32'))
        value = torch.unsqueeze(value, 0)
        print(value)
        if self.values is None:
            self.values = nn.Parameter(value)
        else:
            self.values = nn.Parameter(torch.cat((self.values, value)))