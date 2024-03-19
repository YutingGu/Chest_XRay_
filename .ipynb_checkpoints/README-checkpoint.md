# Chest_XRay_Classification

## Data Download
Run all cells in [Dataset_Download.ipynb](Dataset_Download.ipynb) to download the dataset. Dataset will be downloaded under the the folder `Dataset/images/`. All files end with `.tar.gz.` can be deleted after running all cells to save space. *Please make sure in your .gitignore file, `Dataset/images/` is included to avoid uploading the whole dataset* 

## Label generation and Splite dataset
Change directory to `Utilities`, use terminal to run the [label_generator.py](Utilities/label_generator.py) to generate one-hot encoded labels for train/validation/test set.