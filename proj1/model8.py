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
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        # for param in self.classifier.parameters():
        #     param.requires_grad = False

    def forward(self, x, predict):
        # nezabudunut na sigmoid(x) a reshape((32,32,3))
        if predict is False:
            pred_img = self.prep(x)
            # loss = torch.cosine_similarity(pred_img, x, 0)
            # loss2 = nn.functional.mse_loss(pred_img, x)
            # print(f'Image Loss: {nn.functional.mse_loss(pred_img, x).detach()}')
            # return (1 - loss) + loss2
            # print(x.shape)
            # print(pred_img.shape)
            confidence1 = self.classifier(pred_img)
            confidence2 = self.classifier(x)
            print(f'Confidence1: {confidence1}\tConfidence2:{confidence2}')
            # conf1_loss = nn.functional.mse_loss(confidence1, torch.Tensor([nn.functional.mse_loss(pred_img, x)]))
            # conf1_loss = nn.functional.mse_loss(confidence1,1 - torch.cosine_similarity(pred_img, x, 0))
            conf1_loss = 1/nn.functional.mse_loss(confidence1, 1 - torch.cosine_similarity(pred_img, x, 0))
            # conf2_loss = nn.functional.mse_loss(confidence2, torch.Tensor([1]))
            conf2_loss = 1 - confidence2
            # loss = conf1_loss + conf2_loss
            # loss = torch.softmax(torch.Tensor([conf1_loss, conf2_loss]), 0)
            print(f'Image Loss: {nn.functional.mse_loss(pred_img, x).detach()}')

            return conf1_loss - conf2_loss
            # # return (-loss) + nn.functional.mse_loss(pred_img, x)
            # return loss/2
        else:
            pred_img = self.prep(x)
            return pred_img


