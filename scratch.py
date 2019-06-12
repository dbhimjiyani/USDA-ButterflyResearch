import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
from torchvision import transforms, datasets
import torchvision

x = True

# -------------------------------------------------------------------------------------------

def dataChecker(x):
    sorted = str(input('Hi, is your data sorted into subfolders by categories that you\'re using for image recognition?\nEnter y/n\n'))
    if sorted.lower() == 'y':
        print('Data: Sorted.')
        x = True
        return x
    elif sorted.lower() == 'n':
        print('Data: Unsorted. Creating CSV')
        x = False
        return x
    else:
        print('Invalid input. Try Again.')
        dataChecker(x)

# -------------------------------------------------------------------------------------------

def parse_image_names(image_directory="data", csv_file="data/data.csv", write_csv=True): # <----- look here should you change the csv from the default
    data = []
    image_paths = [fn for fn in os.listdir(image_directory)
                   if os.path.splitext(fn)[-1].lower() in {".jpg", ".jpeg", ".tiff", ".tif"}]

    labels = ["sample_id", "species", "seasonal_variant", "view", "filename"]  # <----- look here should you change the csv from the default
    data = []

    for filename in image_paths:

        match_obj = re.match(r'([a-zA-Z0-9]+)_(smeared|contrasted)*_*([a-zA-Z0-9]+)_([a-zA-Z0-9]+)',  # <---- look here should you change the csv from the default
                             os.path.splitext(filename)[0])
        if match_obj is None:
            # todo: emit log entry
            pass

        # these should exist in all data <----- look here should you change the csv from the default
        species = match_obj.group(1)
        sample_id = match_obj.group(3)
        view = match_obj.group(4)

        # some entries have no seasonal variation
        if match_obj.group(2) is not None:
            seasonal_variant = match_obj.group(2)  # seasonal variations {contrasted, smeared}
        else:
            seasonal_variant = ""

        data.append((sample_id, species, seasonal_variant, view, filename))

    sample_df = pd.DataFrame.from_records(data, columns=labels, index="sample_id")

    # assign integers to each unique species, ordered alphabetically, and create dict
    int_key_dict = {species: sp_int for sp_int, species in enumerate(sorted(sample_df['species'].unique()))}

    # append this new mapping on to the dataframe, so we have the corresponding int value col in the DF now
    sp_ints = sample_df["species"].apply(lambda sp: int_key_dict[sp])

    highest_int = len(int_key_dict) - 1
    sample_df["sp_float"] = sp_ints.apply(lambda sp_int: sp_int / highest_int)

    if write_csv:
        sample_df.to_csv(csv_file)

    return sample_df

# -------------------------------------------------------------------------------------------
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# -------------------------------------------------------------------------------------------

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# -------------------------------------------------------------------------------------------

# This dataset class can now be imported into the rest of the scripts in the software -- yey!
class ButterflyLoader(data.Dataset):
    # make the dataset to point to our image collection
    dataChecker(x)
    def __init__(self, csv_file="data/data.csv", image_dir="data", transform=None):

        if x:
            butterflyDS = datasets.ImageFolder(root='data/train', transform=data_transform)
            loader = torch.utils.data.DataLoader(butterflyDS, shuffle=True, batch_size=4, num_workers=4)
        else:
            self.sample_df = parse_image_names(image_dir, csv_file)
            # transform is optional and can be called on a sample if like
            self.transform = transform
            self.image_dir = image_dir
            loader = ButterflyLoader("data/common/Data.csv", image_dir="data/common")


    def __len__(self):
        return len(self.sample_df)

    # def __getitem__(self, idx):
    #     filename = self.sample_df.iloc[idx, 3]
    #     path = os.path.join(self.image_dir, filename)
    #
    #     image = skimage.io.imread(path)
    #     sp_float = self.sample_df.iloc[idx, 4]
    #
    #     sample = (image, sp_float)
    #
    #     if self.transform:
    #         sample = self.transform(sample)
    #
    #     return sample

# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # quick test
    train_loader = ButterflyLoader()
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images))

