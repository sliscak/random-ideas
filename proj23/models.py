import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW, SGD
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import WikiDataset, WikiDataset_2, WikiDataset_3


class Pavian(LightningModule):

    def __init__(self, output_size=1):  # reset=False
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
            torch.nn.Linear(1000, output_size),
            # torch.nn.Sigmoid(),
            torch.nn.Softmax(),
        )

    def forward(self, x):
        x = self.m(x)
        # print(x)
        return x

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # print(batch)
        # exit()
        x_batch, y_batch = batch
        loss = torch.tensor([0.0])
        for i in range(len(x_batch)):
            x,y = x_batch[i], y_batch[i]
            loss += F.mse_loss(self(x), y)
        # x, y = batch
        # x = x[0]
        # y = y[0]
        # breakpoint()
        # self(x)
        # loss = F.mse_loss(self(x), y)
        # loss = F.mse_loss(self(x_batch), y_batch)
        # loss = F.cross_entropy(self(x_batch), y_batch)
        return pl.TrainResult(loss)

    def train_dataloader(self):
        return DataLoader(WikiDataset(), batch_size=10)
    # def training_step(self, batch, batch_idx):
    #       # x_batch, y_batch = batch
    #       loss = torch.tensor([0], dtype=torch.float)
    #       for b in batch:
    #           x, y = b
    #           y_hat = self(x)
    #           loss += F.mse_loss(y_hat, y)
    #       # loss = F.l1_loss(y_hat, y)
    #       # loss = F.cross_entropy(y_hat, y)
    #       # loss = - kornia.psnr_loss(y_hat, y, 1)
    #       # y_hat_hist = torch.histc(y_hat * 255, bins=255)
    #       # y_hist = torch.histc(y * 255, bins=255)
    #       # loss = F.mse_loss(y_hat_hist, y_hist)
    #       # loss = kornia.ssim
    #       tensorboard_logs = {'train_loss': loss}
    #       return {'loss': loss, 'log': tensorboard_logs}

class Pavian_2(LightningModule):

    def __init__(self, output_size=1):  # reset=False
        super().__init__()
        self.m = torch.nn.Sequential(
            torch.nn.Linear(1, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000 , 1),
            # torch.nn.Sigmoid(),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.m(x)
        return x

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.003)#lr=0.0001)#lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        loss = torch.tensor([0.0])
        for i in range(len(x_batch)):
            x,y = x_batch[i], y_batch[i]
            loss += F.mse_loss(self(x), y)
        print(f'\n{loss.detach()}\n')
        # self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # x, y = batch
        # x = x[0]
        # y = y[0]
        # breakpoint()
        # self(x)
        # loss = F.mse_loss(self(x), y)
        # loss = F.mse_loss(self(x_batch), y_batch)
        # loss = F.cross_entropy(self(x_batch), y_batch)
        return pl.TrainResult(loss)

    # def train_dataloader(self):
    #     return DataLoader(WikiDataset(), batch_size=10, shuffle=True)
    # def training_step(self, batch, batch_idx):
    #       # x_batch, y_batch = batch
    #       loss = torch.tensor([0], dtype=torch.float)
    #       for b in batch:
    #           x, y = b
    #           y_hat = self(x)
    #           loss += F.mse_loss(y_hat, y)
    #       # loss = F.l1_loss(y_hat, y)
    #       # loss = F.cross_entropy(y_hat, y)
    #       # loss = - kornia.psnr_loss(y_hat, y, 1)
    #       # y_hat_hist = torch.histc(y_hat * 255, bins=255)
    #       # y_hist = torch.histc(y * 255, bins=255)
    #       # loss = F.mse_loss(y_hat_hist, y_hist)
    #       # loss = kornia.ssim
    #       tensorboard_logs = {'train_loss': loss}
    #       return {'loss': loss, 'log': tensorboard_logs}

# def init_weights(m):
#     if type(m) == torch.nn.Linear:
#         torch.nn.init.ones_(m.weight)
#         # torch.nn.init.
#         # m.bias.data.fill_(1)

class Pavian_3(LightningModule):

    def __init__(self):  # reset=False
        super().__init__()
        self.m = torch.nn.Sequential(
            torch.nn.Linear(1, 2, bias=True),
            torch.nn.LeakyReLU(),
            # torch.nn.ReLU(),
            # torch.nn.Sigmoid(),
            torch.nn.Linear(2, 1, bias=True),
            torch.nn.Sigmoid(),
        )
        # self.m.apply(init_weights)

    def forward(self, x):
        x = self.m(x)
        return x

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.1)#1e-2)
        # optimizer = SGD(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        # breakpoint()
        x_batch, y_batch = batch
        loss = torch.tensor([0.0])
        for i in range(len(x_batch)):
            x,y = x_batch[i], y_batch[i]
            loss += F.mse_loss(self(x), y)
        print(f'\n{loss.detach()}\n')
        return pl.TrainResult(loss)

    # def train_dataloader(self):
    #     return DataLoader(WikiDataset_3, batch_size=1, shuffle=False)

