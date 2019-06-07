# DINNGuS by USDA-ARS

## Determining Insects using Neural Networks by their Genus and/or Species

## Topics
- [Software Purpose](#software-purpose)
- [Software Structure](#software-structure)
- [Modifying DINNGuS for Your Own Use](#modifying-DINNGuS-foe-your-own-use)
- [About the Devs](#about-the-devs)

## Software Purpose


## Software Structure
* NameExtractor
* TrainTorch


## Modifying DINNGuS for Your Own Use
* Name Extractor: In order to use your own Dataset, you will need to modify the NameExtractor script which extracts vital data from the the filenames and sorts them in to a categories within a Pandas DataFrame. This makes use of RegEx extraction of the species name, insect ID, view of the critter and file name. This RegEx can be modified at https://regex101.com/r/rKpb3M/1

## About the Devs