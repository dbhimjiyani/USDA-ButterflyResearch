from pathlib import Path # Used to change filepaths
import os
import glob
import csv
import re
# Check the directory. Just so we know exactly where we is u kno
print(os.getcwd())

# PLEASE MAKE SURE YOU ARE LINKING TO THE RIGHT DIRECTORY THANK YOU
# A List of images here, we're making it dynamically by grabbing it from a folder called photo
image_paths = [Path(item) for i in [glob.glob(r'data\common\*.%s' % ext) for ext in ["jpg", "jpeg", "tiff", "tif"]] for item in i]
# --------------- look above should you change the csv from the default


def createCSV(path, writer):
    #text = re.sub(r'data\\train\\', '', str(path))
    text = (path.stem)
    # Regex for extracting details from name of the file, such as the species type, code and the angle of shot
    matchObj = re.match(
        r'([a-zA-Z0-9]+)_(smeared|contrasted)*_*([a-zA-Z0-9]+)_([a-zA-Z0-9]+)', text) # <--------------- look here should you change the csv from the default
        # check what this regex is doing by going on https://regex101.com/ & testing against random filenames from your folder

    try:
        species = matchObj.group(1)  # Species # <--------------- look here should you change the csv from the default
        bfId = matchObj.group(3)  # UASMXXXX
        view = matchObj.group(4)  # self-explanatory
    except AttributeError:
        species = "Error_Species"
        bfId = "Error"
        view = "Error"

        # Had to do seasonal separately as the value may be null if the butterflly has had no seasonal variation
    if(matchObj.group(2) is None):
        seasonal = ""
    else:
        seasonal = matchObj.group(2)  # contrasted? smeared? what it iz? (seasonal variations)

    writer.writerow([bfId, species, seasonal, view, path]) # write the newly assigned variables to the file
    print(species + "  " + seasonal + "     " + text) # print formatting no biggie


with open('data/common/Data.csv', mode='w') as bfs: # <--------------- look here should you change the csv from the default
    # opening the csv file and assigning a writer
    bfs_writer = csv.writer(bfs, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # writing row
    bfs_writer.writerow(['Butterfly ID', 'Species', 'Seasonal Variations', 'View', 'FileName'])
    # for loop over image paths so we do the following all images are in the csv
    for img_path in image_paths:
        # call the function written above
        createCSV(Path(img_path), bfs_writer)

print("youe foole. creating the csv my damn self because nobody helps me in this house")


