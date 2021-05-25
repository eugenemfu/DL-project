# Decription

__Team members__: Daria Diatlova, Eugene Sakhno

__Theme__: Image augmentation using image generation with GAN to improve multiclass emotion classification accuracy on images of minor classes and overall testing dataset.

__Dataset__: [FER2013](https://www.kaggle.com/msambare/fer2013). It consists of 28,709 images and the test set consists of 3,589 images labeled by 7 emotions: angry, disgust, fear, happy, sad, neutral, surprise. Dataset has strong class disbalance – for example it’s obvious that images of disgust emotion belong to the minor class. 

Our hypothesis is that accuracy of a simple CNN model for image classification could be improved by generating more images and classes equalization.

### Plan: 

1. Explore:

	- state-of-art CNN models architecture that are used to solve multiclass emotion classification problems,

	- state-of-art GAN models architecture that are used for generating face images.

2. Implement a simple CNN model, train it on the FER2013 dataset and analyze the results.

3. Augment data using GAN, analyze the results of CNN model trained on augmented FER2013 dataset. 

4. Analyze how parameters and architecture of GAN affect CNN model accuracy. 


# How to use

1. Download [dataset](https://www.kaggle.com/msambare/fer2013) and extract zip to `data` folder.

2. Run `make_csv.py`. This will create two csv files (`train.csv` and `test.csv`) with paths and numerical labels of the images.

3. Run `training.ipynb` which trains a CNN model using clean dataset from `train.csv` and saves the model to `cnn.pkl`.

4. To be continued...


# To Do

- Debug training on cuda

- Augmentation

- Reproduce same training on augmented data

- ...