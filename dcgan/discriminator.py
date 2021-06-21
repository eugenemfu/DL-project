import torch.nn as nn
import torch
import torch.optim as optim

from dcgan.utils import GanArgument, weights_init
from dcgan.utils import device


class Discriminator(nn.Module):
    def __init__(self, n_gpu=GanArgument.GPU.value):
        super(Discriminator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(GanArgument.CHANNELS.value,
                      GanArgument.D_FEATURE_MAP_SIZE.value,
                      4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(GanArgument.D_FEATURE_MAP_SIZE.value,
                      GanArgument.D_FEATURE_MAP_SIZE.value * 2,
                      4, 2, 1, bias=False),
            nn.BatchNorm2d(GanArgument.D_FEATURE_MAP_SIZE.value * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(GanArgument.D_FEATURE_MAP_SIZE.value * 2,
                      GanArgument.D_FEATURE_MAP_SIZE.value * 4, 4, 2, 1,
                      bias=False),
            nn.BatchNorm2d(GanArgument.D_FEATURE_MAP_SIZE.value * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(GanArgument.D_FEATURE_MAP_SIZE.value * 4,
                      GanArgument.D_FEATURE_MAP_SIZE.value * 8,
                      4, 2, 1, bias=False),
            nn.BatchNorm2d(GanArgument.D_FEATURE_MAP_SIZE.value * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(GanArgument.D_FEATURE_MAP_SIZE.value * 8,
                      1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_tensor: torch.tensor) -> torch.tensor:
        return self.main(input_tensor)


netD = Discriminator().main.to(device)
netD.apply(weights_init)

optimizer_discriminator = optim.Adam(netD.parameters(),
                                     lr=GanArgument.LEARNING_RATE.value,
                                     betas=(GanArgument.BETTA_1.value, GanArgument.BETTA_2.value))
