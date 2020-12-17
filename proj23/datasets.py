from torch.utils.data import Dataset
import shelve
import torch

class WikiDataset(Dataset):
    """enwik9 dataset."""
    """Either preprocess or process data on the fly"""

    def __init__(self, file_path="enwik9/enwik9"):
      self.data = []
      with shelve.open('db.bin') as db:
          # TODO: rename 'chars' variable to 'table'
          # Chars is a tuple!
          self.chars = db['chars']
          self.table_c2i = {k:v for v,k in enumerate(self.chars)}
          self.table_i2c = {k: v for k,v in enumerate(self.chars)}
          db['table_c2i'] = self.table_c2i
          db['table_i2c'] = self.table_i2c
      self.max_line_len = 0
      # TODO: Sanitize input/output, the last chunk is too short!
      with open(file_path, 'r', encoding="utf-8") as fd:
          chunk_size = 100
          chunk = fd.read(chunk_size)
          self.data.extend(chunk)
          # while len(chunk := fd.read(chunk_size)) > 0:
          #     self.data.extend(chunk)
          print(f'Dataset Len: {len(self.data)}')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # print(self.data[idx])
        x = torch.tensor([
            float(idx / len(self.data))
        ])
        # x = x + 0.0
        char_id = self.table_c2i[self.data[idx]]
        y = torch.zeros((len(self.table_c2i,)))
        y[char_id] = 1.0
        # y = y + 0.0
        # x = torch.tensor([0.5])
        # y = torch.zeros((len(self.table_c2i,)))
        return x, y
        # return self.data[idx]

class WikiDataset_2(Dataset):
    """enwik9 dataset."""
    """Either preprocess or process data on the fly"""

    def __init__(self, file_path="enwik9/enwik9"):
      self.data = []
      with shelve.open('db.bin') as db:
          # TODO: rename 'chars' variable to 'table'
          # Chars is a tuple!
          self.chars = db['chars']
          self.table_c2i = {k:v for v,k in enumerate(self.chars)}
          self.table_i2c = {k: v for k,v in enumerate(self.chars)}
          db['table_c2i'] = self.table_c2i
          db['table_i2c'] = self.table_i2c
      self.max_line_len = 0
      # TODO: Sanitize input/output, the last chunk is too short!
      with open(file_path, 'r', encoding="utf-8") as fd:
          chunk_size = 100
          chunk = fd.read(chunk_size)
          self.data.extend(chunk)
          # while len(chunk := fd.read(chunk_size)) > 0:
          #     self.data.extend(chunk)
          print(f'Dataset Len: {len(self.data)}')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # print(self.data[idx])
        x = torch.tensor([
            float(idx / len(self.data))
        ])
        # x = x + 0.0
        char_id = self.table_c2i[self.data[idx]]
        y = torch.tensor([char_id/int(len(self.chars))])
        # y = y + 0.0
        # x = torch.tensor([0.5])
        # y = torch.zeros((len(self.table_c2i,)))
        return x, y
        # return self.data[idx]

class WikiDataset_3(Dataset):
    def __init__(self):
        self.data = [torch.tensor([0.0]), torch.tensor([0.5]), torch.tensor([0.7])]
        self.data_y = [torch.tensor([0.9]), torch.tensor([0.2]), torch.tensor([0.7])]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.data_y[idx]
        return x, y