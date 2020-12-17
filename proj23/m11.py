# for x in range(100000000000):
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import json
import shelve
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from os import path
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import json
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW
import pytorch_lightning as pl
from os import path
from tqdm import tqdm
from torchsummary import summary
from time import time
from models import Pavian

def build_char_array():
    file_path = "enwik9/enwik9"

    big_array = []
    tstart = time()
    with open(file_path, 'r', encoding="utf-8") as fd:
        chunk_size = 15
        i = 0
        while len(chunk := fd.read(chunk_size)) > 0:
            # if i == 10000:
            #     break
            big_array.append(chunk)
            print(big_array)
            i += 1
    print(big_array)
if __name__ == "__main__":
    build_char_array()