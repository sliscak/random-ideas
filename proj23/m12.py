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
from models import Pavian_2 as Pavian
from datasets import WikiDataset_2

def build_char_array():
    file_path = "enwik9/enwik9"

    big_array = []
    tstart = time()
    with open(file_path, 'r', encoding="utf-8") as fd:
        chunk_size = 1000
        i = 0
        while len(chunk := fd.read(chunk_size)) > 0:
            # if i == 10000:
            #     break
            big_array.extend(chunk)
            i += 1
    tend = time()
    tdiff = tend - tstart
    print(f'Array loaded/build in: {tdiff} seconds')
    # print(d)
    # print(f'LEN:{len(big_array)} with {i} iterations')

    # counters = [Counter() for x in range(5)]
    # print('here')
    result = []
    word_len = 1
    step_size = 1

    print('Starting')
    tstart = time()
    chars = set()
    for x in range(0, len(big_array) - 1, step_size):
        chars.update([''.join(big_array[0 + x:1 + x])])
        if (x % 1000) == 0:
            print(list(chars)[-10:])
    tend = time()
    tdiff = tend - tstart
    print(f'Chars Array(set) build in: {tdiff} seconds')
    # result.extend(chars)

    print(f"C:{x}/{word_len - 1} LEN: {len(chars)} iter: {x}")
    tstart = time()
    with shelve.open('db.bin') as db:
        db['chars'] = chars
        print('Done')
        tend = time()
        tdiff = tend - tstart
        print(f'DB opened and written to in: {tdiff} seconds')

if __name__ == "__main__":
    with shelve.open('db.bin') as db:
        # TODO: rename 'chars' variable to 'table'
        # Chars is a tuple!
        chars = db['chars']
        # db['chars'] = tuple(chars)
    model = Pavian(len(chars))
    print(torch.max(model(torch.tensor([0.25]))))
    trainer = pl.Trainer(gpus=0, max_epochs=15000)# progress_bar_refresh_rate=20)
    train_dataloader = DataLoader(WikiDataset_2(),batch_size=500, shuffle=True)
    trainer.fit(model, train_dataloader=train_dataloader)

    with shelve.open('db.bin') as db:
        # TODO: rename 'chars' variable to 'table'
        # Chars is a tuple!
        chars = db['chars']
        table_c2i = {k: v for v, k in enumerate(chars)}
        table_i2c = {k: v for k, v in enumerate(chars)}
        db['table_c2i'] = table_c2i
        db['table_i2c'] = table_i2c
        dataset = WikiDataset_2()
        for data in dataset:
            x,y = data
            out_orig = model(x)
            out = out_orig * len(chars)
            out = int(out)
            # amax = int(torch.argmax(out))
            print(f'[START\t\nINP: {x} Y_ground_t: {y}\nOUT_orig: {out_orig} \nOUT: {out} \nCHAR: {table_i2c[out]}\n XC: {x*len(dataset)}\n\tEND]')



    # summary(model, (1,))
    # x_val = torch.tensor([1.0])
    # y_out = model(x_val)
    # print(x_val)
    # print(y_out)
    # y_out = model(x_val)
    # amax = torch.argmax(y_out, dim=0)
    # print(f'Index: {amax}, Char: {chars[amax]}')
    # # print(model.)
    # build_char_array()
