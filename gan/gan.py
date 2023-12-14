import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GAN():
    def __init__(self, config, dataloader, device):
        super(GAN, self).__init__()
        # Runtime configs
        self.config = config
        self.device = device
        self.dataloader = dataloader
        # Loss function
        self.loss = torch.nn.BCELoss()
        # Initialize generator and discriminator
        self.generator = Generator(self.config.latent_dim, self.config.img_shape).to(self.device)
        self.discriminator = Discriminator(self.config.img_shape).to(self.device)
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2))

    # ----------
    #  Training
    # ----------
    def train(self):
        for epoch in range(self.config.n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):

                # Adversarial ground truths
                valid = torch.full((imgs.size(0), 1), 1.0, device=self.device, requires_grad=False)
                fake = torch.full((imgs.size(0), 1), 0.0, device=self.device, requires_grad=False)

                # Configure input
                real_imgs = torch.tensor(imgs, device=self.device, dtype=torch.float32)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], self.config.latent_dim)), device=self.device, dtype=torch.float32, requires_grad=True)
                
                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.loss(self.discriminator(real_imgs), valid)
                fake_loss = self.loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.config.n_epochs, i, len(self.dataloader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.config.sample_interval == 0:
                    save_image(gen_imgs.data[:25], f"gan/{self.config.dataset}/%d.png" % batches_done, nrow=5, normalize=True)