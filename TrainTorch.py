import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from NameExtractor import ButterflyDataset
import torch
import os
import torchvision
from skimage import io, transform
from torchvision import transforms, datasets
# import torch.utils.data.Dataset as Dataset


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':

    butterflyDS = datasets.ImageFolder(root='data/train', transform=data_transform)
    train_loader = torch.utils.data.DataLoader(butterflyDS, shuffle=True, batch_size = 4, num_workers = 4)

    testDS = datasets.ImageFolder(root='data/validation')
    test_loader = torch.utils.data.DataLoader(testDS, shuffle=True, batch_size = 4, num_workers=4)

    classes = ('faunus_contrasted', 'faunus_smeared', 'gracilisG', 'gracilisZ', 'progne', 'satyrus_contrasted', 'satyrus_smeared')

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images))


    # # call the object/class
    # loader = ButterflyDataset("data/common/Data.csv", image_dir="data/common")
    #
    # # get some random training images
    # dataiter = iter(loader)
    # sp_float, image = next(dataiter)
    # print(sp_float)
    # plt.imshow(image)
    # plt.show(block=True)

