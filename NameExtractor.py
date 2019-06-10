import os
import re
import skimage

import pandas as pd
from torch.utils import data


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

# This dataset class can now be imported into the rest of the scripts in the software -- yey!
class ButterflyDataset(data.Dataset):
    # make the dataset to point to our image collection

    def __init__(self, csv_file="data/data.csv", image_dir="data", transform=None):

        # if os.path.isfile(csv_file):
        #     self.sample_df = pd.read_csv(csv_file, index_col=0)
        # else:
        self.sample_df = parse_image_names(image_dir, csv_file)

        # transform is optional and can be called on a sample if like
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.sample_df)

    def __getitem__(self, idx):
        filename = self.sample_df.iloc[idx, 3]
        path = os.path.join(self.image_dir, filename)

        image = skimage.io.imread(path)
        sp_float = self.sample_df.iloc[idx, 4]

        sample = (sp_float, image)

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    # quick test
    print(next(iter(ButterflyDataset("data/common/Data.csv", image_dir="data/common"))))
