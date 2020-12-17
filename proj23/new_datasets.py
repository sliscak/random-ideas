from torch.utils.data import Dataset
import shelve
import torch
import numpy as np


class WikiDataset(Dataset):
    """enwik9 dataset."""
    """Either preprocess or process data on the fly"""

    def __init__(self, file_path="enwik9/enwik9", dtype=int):
        self.data = []
        with open(file_path, 'r', encoding="utf-8") as fd:
            chunk_size = 4096
            i = 0
            while chunk := fd.read(chunk_size):
                # chunk = np.array(chunk, dtype=int)
                chunk = [ord(c) for c in chunk]
                # print(np.max(np.array(chunk)))
                # if dtype is float:
                #     chunk = np.array(chunk) + 1
                #     print(np.max(chunk))
                #     chunk = chunk / 255 # max ascii value, was 128
                # else:
                #     # or float whole number
                #     chunk = np.array(chunk, dtype=int)
                self.data.extend(chunk)
                i += 1
                if i >= 1:
                    break
            print(f'Dataset Len: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# dataset = WikiDataset(dtype=int)
# print(np.max(dataset[0]))
# breakpoint()