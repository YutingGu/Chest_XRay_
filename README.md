# Chest_XRay_Classification

## General Instructions
### Step 1 - clone this `Chest_XRay_Classification` GitHub folder
- In Github, create a personal token if not yet done so (Settings -> Developer Settings -> Personal access tokens). This will be used when logging in to GitHub through the git clone command.
- Clone this `Chest_XRay_Classification` GitHub folder on local device, using
```
git clone https://github.com/YutingGu/Chest_XRay_Classification.git
```
- You will be asked to enter the password to log in, which is your personal token.
- After logging in, the GitHub folder will be successfully clone to your local device.


### Step 2 - Image Data Download
The data used for this project includes image data and label data. Since image data have not been uploaded to GitHub, we need a further step to download the image data to local device.
- Run all cells in [Dataset_Download.ipynb](Dataset_Process_Scripts/Dataset_Download.ipynb) to download and unzip the image dataset.

**Note:**
* Dataset will be downloaded under the the folder `Dataset/images/`. All files end with `.tar.gz.` can be deleted after running all cells to save space, these are the compressed image files.
* Please make sure in your [.gitignore](.gitignore) file, `Dataset/images/` is included to avoid uploading the whole dataset

#### The 'Dataset' folder
This folder contains the original data downloaded from https://paperswithcode.com/dataset/chestx-ray8, and the processed data (one-hot encoded labels for train/validation/test set) after running `Utilities/label_generator.py`.

**The original data:**
- `CXR8_Data_Entry_2017.csv`, `test_list.txt`, `train_val_list.txt`

**Processed data:**
- `train_list.txt`, `val_list.txt`,
- `test_label.csv`, `train_label.csv`, `val_label.csv`

**The following is the method for doing one-hot encoded label generation and dataset split:**
* Change directory to `Utilities`, use terminal to run the [label_generator.py](Utilities/label_generator.py) to generate one-hot encoded labels for train/validation/test set.



