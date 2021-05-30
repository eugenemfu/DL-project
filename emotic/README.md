# EMOTIC dataset

Dataset consists of images collected from several datasets – source images can be downloaded from aws via the [link](https://hse-ml.s3-eu-west-1.amazonaws.com/emotic.zip).
Dataset is divided by authors and labels can be found in `Annotations.mat` file (can be downloaded via the [link here](https://hse-ml.s3-eu-west-1.amazonaws.com/Annotations.zip)). 

## Preprocessing
### Step-1
Run `label_dataset.py` script from command line with 2 obligatory arguments


Argument | Description
--- | ---
|**Annotations.mat**| Absolute path to the Annotations.mat file.|
|**holdout**| Part of dataset you are intended to label. Available values are: `train`, `val`, `test`.|

Output files: `train.csv`, `val.csv`, `test.csv` files that will be created in the `Annotations.mat` parent directory. Labeled
csv files can be found in the [repository dir](https://github.com/eugenemfu/DL-project/tree/main/emotic/labeled_datset). 

### Step-2
Each image has 1 or 2 emotions correspond to it. To build `multilabel` classification we will need one-hot-encoded labels
that can be found in the [repository dir](https://github.com/eugenemfu/DL-project/tree/main/emotic/encoded_labels). Each image has
27 target classes: 26 emotions and one `No emotion` class added to substitute `nan` values. Keys to the emotion codes can be found [here](https://github.com/eugenemfu/DL-project/blob/main/emotic/encoded_labels/emotion_keys.csv).


### Step-3
To group images by `train`, `val` and `test` holdout images – create `output_folder` and run 
`cp -R dataset_folder/. output_folder/ ` where `dataset_folder` correspond to the image folder [emotic.zip]('https://hse-ml.s3-eu-west-1.amazonaws.com/emotic.zip).

When all images from emotic dataset are in the `output_folder` run [group_by_omages.py](https://github.com/eugenemfu/DL-project/blob/main/emotic/preprocessing/group_by_images.py) script from the command line with the following arguments:

Argument | Description
--- | ---
|**folder_path**| Absolute path to the folder contained all Emotic images.|
|**target_path**| Path to the csv file get by [label_dataset.py](https://github.com/eugenemfu/DL-project/blob/main/emotic/preprocessing/label_dataset.py) script.|
|**holdout**| Part of dataset you are intended to label. Available values are: `train`, `val`, `test`. |

As the result `train`, `val` and `test` folders with images will be created in the parent directory of `folder_path`.

Alternitevly, folders with images can be downloaded from aws: 

– [`train`](https://hse-ml.s3-eu-west-1.amazonaws.com/test.zip) (17 077) images

– [`val`](https://hse-ml.s3-eu-west-1.amazonaws.com/val.zip) (2 088 images)

– [`test`](https://hse-ml.s3-eu-west-1.amazonaws.com/test.zip) (4 389 images)
