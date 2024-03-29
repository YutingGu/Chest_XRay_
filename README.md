# Chest_XRay_Classification

## General Instructions
### Step 1 - clone this `Chest_XRay_Classification` GitHub folder
- In Github, create a personal token if not yet done so (Settings -> Developer Settings -> Personal access tokens). This will be used when logging in to GitHub through the git clone command.
- Login to Imperial HPC and create a personal folder that will be used for this ML project.
- Clone this HPC folder to the `Chest_XRay_Classification` GitHub folder, using
```
git clone https://github.com/YutingGu/Chest_XRay_Classification.git
```
- You will be asked to enter the password to log in, which is your personal token.
- After logging in, the GitHub folder will be successfully clone to your HPC folder.


### Step 2 - Image Data Download
The data used for this project includes image data and label data. Since image data have not been uploaded to GitHub, we need a further step to download the image data to HPC.
- Run all cells in [Dataset_Download.ipynb](Dataset_Download.ipynb) to download the dataset.
<br />
Dataset will be downloaded under the the folder `Dataset/images/`. All files end with `.tar.gz.` can be deleted after running all cells to save space.
<br />
*Please make sure in your [.gitignore](.gitignore) file, `Dataset/images/` is included to avoid uploading the whole dataset*

## Folder Explanation
### The 'Dataset' folder
This folder contains the original data downloaded from https://paperswithcode.com/dataset/chestx-ray8, and the processed data (one-hot encoded labels for train/validation/test set) after running [label_generator.py](Utilities/label_generator.py).

**The original data:**
- `CXR8_Data_Entry_2017.csv`, `test_list.txt`, `train_val_list.txt`

**Processed data:**
- `train_list.txt`, `val_list.txt`,
- `test_label.csv`, `train_label.csv`, `val_label.csv`
<br />
The following is the method for doing one-hot encoded label generation and dataset split:
<br />
Change directory to `Utilities`, use terminal to run the [label_generator.py](Utilities/label_generator.py) to generate one-hot encoded labels for train/validation/test set.



