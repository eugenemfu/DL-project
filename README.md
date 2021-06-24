# Decription

__Team members__: Daria Diatlova, Eugene Sakhno

__Theme__: Image augmentation using image generation with GAN to improve multiclass emotion classification accuracy on images of minor classes and overall testing dataset.

__Dataset__: [FER13](https://datarepository.wolframcloud.com/resources/FER-2013) consists of 35,886 images. 

<img src = resources/emotion_samples/image_samples.jpg width="400" height="400">

The task is multiclassification. Our hypothesis is that accuracy of a simple CNN model for image classification could be improved by generating more images and classes equalization.

### Plan: 

1. Explore:

	- state-of-art CNN models architecture that are used to solve multiclass emotion classification problems,

	- state-of-art GAN models architecture that are used for generating face images.

2. Implement a simple CNN model, train and analyze the results.

3. Augment data using GAN, analyze the results of CNN model trained on augmented dataset. 

4. Analyze how parameters and architecture of GAN affect CNN model accuracy. 


# Data overview
There are 7 image classes in the dataset that are encoded as following:

**Emotion** | Angry | Disgust | Fear | Happy | Neutral | Sad | Surprise
---|---|---|---|---|---|---|---
Id | 0| 1 | 2 | 3 | 4 | 5 | 6 
Image | <img src = resources/emotion_samples/0.png width="64" height="64"> | <img src = resources/emotion_samples/1.png width="64" height="64"> | <img src = resources/emotion_samples/2.png width="64" height="64"> | <img src = resources/emotion_samples/3.png width="64" height="64"> | <img src = resources/emotion_samples/4.png width="64" height="64"> | <img src = resources/emotion_samples/5.png width="64" height="64"> | <img src = resources/emotion_samples/6.png width="64" height="64"> |

Classes are unbalanced and distributed as following:

<img src = resources/plots/emotion.png width="700" height="400">

# How to preprocess dataset
1. Download [`data.zip`](https://www.kaggle.com/msambare/fer2013?select=test), unarchive it and put all content in `data`
folder in the root of repository. Run [`make_csv.py`](make_csv.py) images to get `csv` filed with labels.There are 7 classes of emotions in the dataset, labels
   Output files will be saved to the `data` directory and would have 2 columns: `path` and `label`. 
2. Run [`resize_data.py`](resize_data.py) script to create `resized_data` folder in the project root. There images will be saved in the format: `64 x 64`.


# How to run classifier
Create `models` folder in the prject root and run `train.py`(train.py) script to save trained model. Model will be trainned apllying [differential augmentation](augment.py):

- brightness shifting;
- transformations;
- crops. 

# How to evaluate classifier

Run `evaluation.py` script to get metrics in the following format:

	Overall accuracy:  0.589
	Overall loss:      1.158
	Class: 0,  share: 0.133,  FR rate: 0.476,  FA rate: 0.644,  F1 score: 0.483
	Class: 1,  share: 0.015,  FR rate: 0.640,  FA rate: 0.135,  F1 score: 0.482
	Class: 2,  share: 0.143,  FR rate: 0.672,  FA rate: 0.337,  F1 score: 0.394
	Class: 3,  share: 0.247,  FR rate: 0.167,  FA rate: 0.262,  F1 score: 0.795
	Class: 4,  share: 0.172,  FR rate: 0.500,  FA rate: 0.376,  F1 score: 0.533
	Class: 5,  share: 0.174,  FR rate: 0.514,  FA rate: 0.624,  F1 score: 0.461
	Class: 6,  share: 0.116,  FR rate: 0.218,  FA rate: 0.323,  F1 score: 0.743


Where,
- `FR`: false reject rate
- `FA`: false alarm rate
- `F1`: f1-score 

