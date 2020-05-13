import torch
import cv2

from proj2.model2 import  Fusher, Cleaner

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('cleaner', cv2.WINDOW_NORMAL)
cleaner = Cleaner()

optimizer = torch.optim.Adam(params=cleaner.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(params=cleaner.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
b = 0.001
imgs = [torch.rand(size=(32, 32, 3)) for i in range(1000)]
i = 0
while True:
    image = imgs[i]
    i += 1
    if i >= len(imgs):
        i = 0
    cleaned_image = cleaner(image)
    loss = criterion(cleaned_image, image)
    # loss = 1 - torch.cosine_similarity(torch.flatten(cleaned_image), torch.flatten(image), dim=0)
    # print(loss)
    # p = torch.sigmoid(cleaner.p)
    p = torch.abs(cleaner.p)
    # loss = torch.abs(loss - (b+p)) + (b+p)
    orig_loss = loss.detach()
    loss = torch.abs(loss - b) + b
    # if loss < 0.0001:
    #     loss = -loss
    optimizer.zero_grad()
    loss.backward()
    print(f'Loss: {orig_loss}\tP: {cleaner.p.detach()}\tB: {b}\tB+P: {b+p.detach()}')
    optimizer.step()
    # image = ((image < 0.25) + (image > 0.75)) * image
    cv2.imshow('image',image.numpy())
    cv2.imshow('cleaner', cleaned_image.detach().numpy())
    cv2.waitKey(100)