# Decription

__Team members__: Daria Diatlova, Eugene Sakhno

__Theme__: Image augmentation using image generation with GAN to improve multiclass emotion classification accuracy on images of minor classes and overall testing dataset.

__Dataset__: [EMOTIC](https://github.com/rkosti/emotic) consists of 23,554 images. 

The task is multilabel classification. Each image has 1 or 2 corresponded emotion. Our hypothesis is that accuracy of a simple CNN model for image classification could be improved by generating more images and classes equalization.

### Plan: 

1. Explore:

	- state-of-art CNN models architecture that are used to solve multiclass emotion classification problems,

	- state-of-art GAN models architecture that are used for generating face images.

2. Implement a simple CNN model, train it on the EMOTIC dataset and analyze the results.

3. Augment data using GAN, analyze the results of CNN model trained on augmented EMOTIC dataset. 

4. Analyze how parameters and architecture of GAN affect CNN model accuracy. 


# How to preprocess dataset

1. Fill the form to access [dataset](https://docs.google.com/forms/d/e/1FAIpQLScXwxhEZu7RpHwgiRqVfb09GzHSSyIm64hJQMgHSLm75ltsFQ/viewform). 
   Download dataset and put `emotic` folder in the root of the project with `Annotation.py` file.
   
2. There are two possible options:

   -  Run `label_dataset.py` script from the command line.

   Command to run from the repository root: 
   
   `python3 -m preprocessing.label_dataset [local_path_to_the_repository]/emotic/Annotations.mat`

	After running the script 4 files will be created in the `emotic` folder: `emotion_keys.csv`, `test.csv`, `val.csv`, `train.csv`.

   - Access labeled files and emotion keys from [repository directory](https://github.com/eugenemfu/DL-project/tree/main/labels).
	
# To Do

- Debug training on cuda

- Augmentation

- Reproduce same training on augmented data

- ...
