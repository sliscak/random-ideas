# Program for the HutterPrize Competition

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import json
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

class PythonDataset(Dataset):
    """Python dataset."""
    """Either preprocess or process data on the fly"""

    def __init__(self, file_path="enwik9/enwik9"):
      self.data = []
      self.max_line_len = 0
      # TODO: Sanitize input/output, the last chunk is too short!
      max_int = 0
      with open(file_path, 'r', encoding="utf-8") as fd:
          while len(chunk := fd.read(50)) > 0:
              self.data.append(chunk)
              # for c in chunk:
              #   c_int = ord(c)
              #   if max_int < c_int:
              #       max_int = c_int
              #       print(f'MAX INT: {max_int}')
              if len(chunk) > self.max_line_len:
                    self.max_line_len = len(chunk)
                    # print(chunk)
          print(f'Dataset Len: {len(self.data)}')
      #     for line in tqdm(fd, desc='Dataset Loading'):
      #       self.data.append(line)
      #       if len(line) > self.max_line_len:
      #           self.max_line_len = len(line)
      #           print(line)
      # print(f'Max Line Len: {self.max_line_len}')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # return self.data[idx]
        return self.data[0]

class Pavian(LightningModule):

  def __init__(self): # reset=False
    super().__init__()
    self.m = torch.nn.Sequential(
        torch.nn.Linear(1, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1),
        torch.nn.ReLU(),
    )

  def forward(self, x):
    x = self.m(x)
    return x

  def configure_optimizers(self):
        optimizer = AdamW(model.parameters(), lr=1e-2)
        return optimizer

  def training_step(self, batch, batch_idx):
        # x_batch, y_batch = batch
        loss = torch.tensor([0], dtype=torch.float)
        for b in batch:
            x, y = b
            y_hat = self(x)
            loss += F.mse_loss(y_hat, y)
        # loss = F.l1_loss(y_hat, y)
        # loss = F.cross_entropy(y_hat, y)
        # loss = - kornia.psnr_loss(y_hat, y, 1)
        # y_hat_hist = torch.histc(y_hat * 255, bins=255)
        # y_hist = torch.histc(y * 255, bins=255)
        # loss = F.mse_loss(y_hat_hist, y_hist)
        # loss = kornia.ssim
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}




if __name__ == '__main__':
    dataset = PythonDataset()
    # print(dataset[0])

    model = Pavian()

    config = {
        # 'gpus': 1, # uncomment to use GPU
        # 'default_root_dir': '/content/drive/My Drive/research/checkpoints',
        'checkpoint_callback': False,
        'max_epochs': 1,
        # 'precision':16,
    }

    def collate_fn(batches):
        with torch.no_grad():
            x_batch = []
            y_batch = []
            l = len(batches[0])
            for e in list(enumerate(batches[0])):
                x_batch.append(torch.tensor([e[0] / (l-1)], dtype=torch.float))
                y_batch.append(torch.tensor([ord(e[1])/1000], dtype=torch.float))
            return zip(x_batch, y_batch)
            # print(x_batch, y_batch)
            # text_batch = ['<mediawiki xmlns="http://www.mediawiki.org/xml/exp']
            # batch = torch.tensor([text_batch])
            # y_batch = torch.tensor([ord(c) for c in [batch for batch in batches]])


            # y_batch = torch.tensor([ord(c) for c in batches[0]])
            # return (torch.tensor(x_batch), torch.tensor(y_batch))


    train_loader = DataLoader(PythonDataset(), batch_size=2, shuffle=False, collate_fn=collate_fn)

    trainer = pl.Trainer(**config)
    trainer.fit(model, train_loader)

