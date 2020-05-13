import torch
from torch import nn

class Net(nn.Module):
    # input is 64 x 64 x 3 image
    def __init__(self):
        super(Net, self).__init__()
        self.fusher = nn.Sequential(
            nn.Linear(32 * 32 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 32 * 32 * 3),
            nn.Sigmoid()
        )

        self.cleaner = nn.Sequential(
            nn.Linear(32 * 32 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 32 * 32 * 3),
            nn.Sigmoid()
        )
        self.p = nn.Parameter(torch.Tensor(1))
        self.b = 0.001
        # for param in self.classifier.parameters():
        #     param.requires_grad = False

    def PSNR(self, mse):
        psnr = -10 * torch.log10(1 / mse)
        return psnr

    def forward(self, x, predict):
        # nezabudunut na sigmoid(x) a reshape((32,32,3))
        if predict is False:
            img = self.cleaner(x)
            loss = nn.functional.mse_loss(img, x)
            loss = (loss - (self.b + abs(self.p))).abs() + (self.b + abs(self.p))
            # loss = (loss - self.b).abs() + self.b
            # print(f'TEST ON RAND: Conf: {c}\tIMG-SIM Loss: {1 - nn.functional.cosine_similarity(rnd, x, dim=0)}')

            return loss #if loss > 0.001 else - (loss * 0.9)
            # feature1 = self.siamese(pred_img)
            # feature2 = self.siamese(x)
            # sim = torch.cosine_similarity(feature1, feature2, 0)
            # # loss = nn.functional.mse_loss(sim, torch.cosine_similarity(pred_img, x, 0))
            # return 1-sim#loss**loss#+(nn.functional.mse_loss(pred_img, x)**nn.functional.mse_loss(pred_img, x))
            # # return (-loss) + nn.functional.mse_loss(pred_img, x)
            # return loss/2
        else:
            cleaned = self.cleaner(x)
            return cleaned


