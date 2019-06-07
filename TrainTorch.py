import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torchvision
from skimage import io, transform
import torchvision.transforms as transforms
# import torch.utils.data.Dataset as Dataset

class ButterflyDataset(torch.utils.data.Dataset):
    # make the dataset to point to our image collection

    def __init__(self, csv_file, root_dir=".", transform=None):
        # transform is optional and can be called on a sample if like
        self.transform = transform
        self.root_dir = root_dir

        self.sample_df = pd.read_csv(csv_file, index_col=0)

        # order all species alphabetically, then assign unique integers to each new type of species
        # and define a map between species name and integer
        int_key_dict = {species: sp_int for sp_int, species in enumerate(sorted(self.sample_df['Species'].unique()))}

        # append this new mapping on to the dataframe, so we have the corresponding int value col in the DF now
        self.sample_df["sp_int"] = self.sample_df["Species"].apply(lambda sp: int_key_dict[sp])

        highest_int = len(int_key_dict) - 1
        self.sample_df["sp_float"] = self.sample_df["sp_int"].apply(lambda sp_int: sp_int/highest_int)


    def __len__(self):
        return len(self.sample_df)

    def __getitem__(self, idx):
        filename = self.sample_df.iloc[idx, 3]
        img_name = os.path.join(self.root_dir, filename)

        image = io.imread(img_name)
        # sample = {'image': image, 'sp_float': self.sample_df.iloc[idx, 5]}
        idx_sp_float = self.sample_df.iloc[idx, 5]
        sample = (idx_sp_float, image)

        if self.transform:
            sample = self.transform(sample)

        return sample


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':

    # call the object/class
    loader = ButterflyDataset("data/common/Data.csv")

    # get some random training images
    dataiter = iter(loader)
    sp_float, image = next(dataiter)
    print(sp_float)
    print(image)

