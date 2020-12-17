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
from models import Pavian_3 as Pavian
from datasets import WikiDataset_3 as WikiDataset

if __name__ == "__main__":
    model = Pavian()
    trainer = pl.Trainer(gpus=0, max_epochs=1000)# progress_bar_refresh_rate=20)
    train_dataloader = DataLoader(WikiDataset(), batch_size=10)
    trainer.fit(model, train_dataloader=train_dataloader)

    dataset = WikiDataset()

    for data in dataset:
        x,y = data
        out = model(x)
        print(f'[START\t\nINP: {x} OUT: {out.detach()} \nY: {y}\n\tEND]')
    print(list(model.parameters()))