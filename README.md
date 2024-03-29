# Chest_XRay_Classification

## The 'Dataset' folder
This folder contains the original data downloaded from https://paperswithcode.com/dataset/chestx-ray8, and the processed data (one-hot encoded labels for train/validation/test set) after running [label_generator.py](Utilities/label_generator.py).

**The original data:**
- `CXR8_Data_Entry_2017.csv`, `test_list.txt`, `train_val_list.txt`

**Processed data:**
- `train_list.txt`, `val_list.txt`,
- `test_label.csv`, `train_label.csv`, `val_label.csv`

## Data Download
Run all cells in [Dataset_Download.ipynb](Dataset_Download.ipynb) to download the dataset. Dataset will be downloaded under the the folder `Dataset/images/`. All files end with `.tar.gz.` can be deleted after running all cells to save space. *Please make sure in your [.gitignore](.gitignore) file, `Dataset/images/` is included to avoid uploading the whole dataset* 

## Label generation and Splite dataset
Change directory to `Utilities`, use terminal to run the [label_generator.py](Utilities/label_generator.py) to generate one-hot encoded labels for train/validation/test set.
