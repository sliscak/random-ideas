import torch
from torch import nn

class Net(nn.Module):
    # input is 64 x 64 x 3 image
    def __init__(self):
        super(Net, self).__init__()
        self.prep = nn.Sequential(
            nn.Linear(32 * 32 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 32 * 32 * 3),
            nn.Sigmoid()
        )

        self.siamese = nn.Sequential(
            nn.Linear(32 * 32 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100))

        self.classifier = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        self.b = nn.Parameter(torch.Tensor([1]))
        # for param in self.classifier.parameters():
        #     param.requires_grad = False

    def PSNR(self, mse):
        psnr = -10 * torch.log10(1 / mse)
        return psnr

    def forward(self, x, predict):
        # nezabudunut na sigmoid(x) a reshape((32,32,3))
        if predict is False:
            pred_img = self.prep(x)
            loss = 1 - nn.functional.cosine_similarity(pred_img, x, dim=0)
            psnr = self.PSNR(loss)
            mse = nn.functional.mse_loss(pred_img, x)
            print(f'CS-LOSS: {loss}\tMSE-LOSS: {mse.detach()}')
            # return mse
            return (mse - 0.00003).abs() + 0.00003
            # return mse if loss > 0.00003 else -loss
            # return (mse - torch.sigmoid(self.b)).abs() + torch.sigmoid(self.b)
            # feature1 = self.siamese(pred_img)
            # feature2 = self.siamese(x)
            # sim = torch.cosine_similarity(feature1, feature2, 0)
            # # loss = nn.functional.mse_loss(sim, torch.cosine_similarity(pred_img, x, 0))
            # return 1-sim#loss**loss#+(nn.functional.mse_loss(pred_img, x)**nn.functional.mse_loss(pred_img, x))
            # # return (-loss) + nn.functional.mse_loss(pred_img, x)
            # return loss/2
        else:
            pred_img = self.prep(x)
            return pred_img


