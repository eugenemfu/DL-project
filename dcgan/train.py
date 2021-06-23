from __future__ import print_function
import time
import random
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imagedataset import ImageDataset
from dcgan.utils import GanArgument, device, plot_loss_with_plt, weights_init
from dcgan.generator import Generator
from dcgan.discriminator import Discriminator
from augment import augment


SEED = 999
# SEED = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def main():
    netG = Generator().main.to(device)
    netG.apply(weights_init)
    optimizer_generator = optim.Adam(netG.parameters(),
                                     lr=GanArgument.LEARNING_RATE.value,
                                     betas=(GanArgument.BETA_1.value, GanArgument.BETA_2.value))

    netD = Discriminator().main.to(device)
    netD.apply(weights_init)
    optimizer_discriminator = optim.Adam(netD.parameters(),
                                         lr=GanArgument.LEARNING_RATE.value,
                                         betas=(GanArgument.BETA_1.value, GanArgument.BETA_2.value))

    real_label = 1.
    fake_label = 0.

    img_list = []

    generator_losses = []
    discriminator_losses = []

    iterations = 0

    # Use resized images to create dataset
    dataset = ImageDataset(pd.read_csv('data64/train.csv'), 1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=GanArgument.BATCH_SIZE.value, shuffle=True)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, GanArgument.LATENT_SIZE.value, 1, 1, device=device)

    print("Starting Training Loop...")

    for epoch in range(GanArgument.EPOCHS.value):
        for i, data in enumerate(dataloader, 0):
            # Update D network: maximize log(D(x)) + log(1 - D(G(z))) and train with real data
            netD.zero_grad()
            real = augment(data['image'].to(device))
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            # D_x = output.mean().item()

            # Train with fake-data and generate batch of latent vectors
            noise = torch.randn(b_size, GanArgument.LATENT_SIZE.value, 1, 1, device=device)
            # Generate fake image batch with G
            fake = augment(netG(noise))
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            # D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_discriminator.step()

            # Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            errG.backward()
            # D_G_z2 = output.mean().item()
            optimizer_generator.step()

            # Save Losses for plotting later
            generator_losses.append(errG.item())
            discriminator_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if i == len(dataloader) - 1:
                print('[%d/%d]  Loss_D: %.4f  Loss_G: %.4f' % (epoch + 1, GanArgument.EPOCHS.value,
                                                               np.mean(discriminator_losses[-len(dataloader):]),
                                                               np.mean(generator_losses[-len(dataloader):])))

                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                filename = f'generators/G_{int(time.time()) % 10000000}.pkl'
                torch.save(netG, filename)
                print(f'Saved generator as {filename}')
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
                plt.show()

                if epoch == GanArgument.EPOCHS.value - 1:
                    plot_loss_with_plt(generator_losses, discriminator_losses)

            iterations += 1


if __name__ == "__main__":
    main()
