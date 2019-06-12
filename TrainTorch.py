import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from NameExtractor import ButterflyDataset
import torch
import os
import torchvision
import torch.optim as optim
from skimage import io, transform
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------------------------

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# -------------------------------------------------------------------------------------------

data_transform = transforms.Compose([
        transforms.Scale((100,100)),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# -------------------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #TODO what dis? play with these numbers cuz like hmmm
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # [self.fc1] This needs to be customized by finding the sqroot of of image size/4(batch size)/16(channels)
        # that makes x view height and width = approx 53 (our size is 179,776)
        self.fc1 = nn.Linear(16 * 53*53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # [x.view] This needs to be customized by finding the sqroot of image size/4(batch size)/16(channels)
        #that makes x view height and width = approx 53 (our size is 179,776)
        x = x.view(x.size(0), 16*53*53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------------------------------------------------------------------------

if __name__ == '__main__':

    butterflyDS = datasets.ImageFolder(root='data/train', transform=data_transform)
    train_loader = torch.utils.data.DataLoader(butterflyDS, shuffle=True, batch_size = 4, num_workers = 0)

    testDS = datasets.ImageFolder(root='data/validation', transform=data_transform)
    test_loader = torch.utils.data.DataLoader(testDS, shuffle=True, batch_size = 4, num_workers=0)

    classes = ('faunus_contrasted', 'faunus_smeared', 'gracilisG', 'gracilisZ', 'progne', 'satyrus_contrasted', 'satyrus_smeared')


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):
    # loop over dataset many times with many variation of the images
    ''' We do this as a part of Data Augmentation, so that machine thinks
     that we are training it on new images everytime. But catching, photographing and
     running DNA sequences on butterflies is hard and time -consuming and we are
     data-poor so we trick "more" data. It also helps prevent overflitting :D '''

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad() #clear previous gradients
        outputs = net(inputs) #TODO forward + backward input
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() # perform updates using calculated gradients

        # print statistics
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

        print('')

print('Finished Training')


dataiter = iter(train_loader)
images, labels = dataiter.next()
print('Forrealz: ' , ' '.join('%5s' % classes[labels[j]] for j in range(4)))
imshow(torchvision.utils.make_grid(images))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(7))
class_total = list(0. for i in range(7))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

    # # call the object/class
    # loader = ButterflyDataset("data/common/Data.csv", image_dir="data/common")
    #
    # # get some random training images
    # dataiter = iter(loader)
    # sp_float, image = next(dataiter)
    # print(sp_float)
    # plt.imshow(image)
    # plt.show(block=True)

