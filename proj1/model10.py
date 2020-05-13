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
        # for param in self.classifier.parameters():
        #     param.requires_grad = False

    def PSNR(self, mse):
        psnr = -10 * torch.log10(1 / mse)
        return psnr

    def forward(self, x, predict):
        # nezabudunut na sigmoid(x) a reshape((32,32,3))
        if predict is False:
            pred_img = self.prep(x) #+ ((torch.rand_like(x) * 0.3)*2)-0.3)
            siam1 = self.siamese(pred_img)
            siam2 = self.siamese(x)
            # sim = 1 - nn.functional.cosine_similarity(siam1, siam2, dim=0)
            sim = (siam1 - siam2).abs()
            conf = self.classifier(sim)
            img_sim = 1 - nn.functional.cosine_similarity(pred_img, x, dim=0)
            mse = nn.functional.mse_loss(conf, img_sim)
            # psnr = self.PSNR(loss)
            mse = nn.functional.mse_loss(pred_img, x)
            # print(f'CS-LOSS: {sim}\tMSE-LOSS: {mse.detach()}')
            print(f'Confidence: {conf}\tImage-Sim Loss: {img_sim}\tMSE-Loss: {mse}')
            rnd = torch.rand_like(x)
            s1 = self.siamese(rnd)
            s2 = self.siamese(x)
            s = (s1 - s2).abs()
            c = self.classifier(s)
            print(f'TEST ON RAND: Conf: {c}\tIMG-SIM Loss: {1 - nn.functional.cosine_similarity(rnd, x, dim=0)}')

            return mse
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


