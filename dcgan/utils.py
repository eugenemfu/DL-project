import torch.nn as nn
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.utils as vutils

from enum import Enum
from typing import NoReturn, List
from torch.utils.data import DataLoader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GanArgument(Enum):
    # Batch size during training
    BATCH_SIZE = 64
    # Spatial size of training images. All images will be resized to this size using a transformer.
    IMAGE_SIZE = 64
    # Number of channels in the training images. For color images this is 3
    CHANNELS = 1
    # Size of z latent vector (i.e. size of generator input)
    LATENT_SIZE = 100
    # Size of feature maps in generator
    G_FEATURE_MAP_SIZE = 64
    # # Size of feature maps in discriminator
    D_FEATURE_MAP_SIZE = 64
    # Number of training epochs
    EPOCHS = 15
    # Learning rate for optimizers
    LEARNING_RATE = 2e-4
    # Beta1 hyperparam for Adam optimizers
    BETA_1 = 0.5
    # Number of GPUs available. Use 0 for CPU mode.
    GPU = 0
    # Betta 2 hyperparam for Adam optimizers
    BETA_2 = 0.999


device = torch.device("cuda:0" if (torch.cuda.is_available() and GanArgument.GPU.value > 0) else "cpu")


def plot_loss_plotly(generator_losses: List, discriminator_losses: List) -> NoReturn:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, len(generator_losses)),
                             y=generator_losses,
                             mode='lines',
                             name='Generator'))
    fig.add_trace(go.Scatter(x=np.arange(0, len(discriminator_losses)),
                             y=discriminator_losses,
                             mode='lines',
                             name='Discriminator'))
    fig.update_layout(title='Loss convergence',
                      xaxis_title='Iterations',
                      yaxis_title='Loss')
    fig.show()


def plot_training_images(dataloader: DataLoader) -> NoReturn:
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:64],
                                             padding=2,
                                             normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def plot_loss_with_plt(generator_losses: List, discriminator_losses: List):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_losses, label="G", )
    plt.plot(discriminator_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
