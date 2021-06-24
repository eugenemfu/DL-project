## Adaptation of Data-Efficient Style GAN for image generation

We've applied [differential augmentation](https://arxiv.org/abs/2006.10738) 
to produce images of minor class:

Real images of disgust emotion

<img src = resources/emotion_samples/real_disgust.png>:


Fake images of disgust emotion 

<img src = resources/emotion_samples/fake_disgust_300.png>:

For training generator and image generation, the adopted version of 
[DiffAugment-stylegan2-pytorch](https://github.com/dariadiatlova/data-efficient-gans/tree/grey-scale) was used.
Examples of training and generating scripts are in the [`train.py`](style_gan/train.py) and [`generate.py`](style_gan/generate.py)
are in the repository notebooks. 




