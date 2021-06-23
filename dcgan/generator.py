import torch.nn as nn
import torch

from dcgan.utils import GanArgument


class Generator(nn.Module):
    def __init__(self, n_gpu=GanArgument.GPU.value):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(GanArgument.LATENT_SIZE.value,
                               GanArgument.G_FEATURE_MAP_SIZE.value * 8, 4, 1, 0,
                               bias=False),
            nn.BatchNorm2d(GanArgument.G_FEATURE_MAP_SIZE.value * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(GanArgument.G_FEATURE_MAP_SIZE.value * 8,
                               GanArgument.G_FEATURE_MAP_SIZE.value * 4, 4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(GanArgument.G_FEATURE_MAP_SIZE.value * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(GanArgument.G_FEATURE_MAP_SIZE.value * 4,
                               GanArgument.G_FEATURE_MAP_SIZE.value * 2, 4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(GanArgument.G_FEATURE_MAP_SIZE.value * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(GanArgument.G_FEATURE_MAP_SIZE.value * 2,
                               GanArgument.G_FEATURE_MAP_SIZE.value, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GanArgument.G_FEATURE_MAP_SIZE.value),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(GanArgument.G_FEATURE_MAP_SIZE.value,
                               GanArgument.CHANNELS.value, 4, 2, 1,
                               bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input_tensor: torch.tensor) -> torch.tensor:
        return self.main(input_tensor)


