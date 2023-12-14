import numpy as np
import time

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
        img = img.view(img.shape[0], *self.img_shape)
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
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class WGAN():
    def __init__(self, config, dataloader, device):
        super(WGAN, self).__init__()
        # Runtime configs
        self.config = config
        self.device = device
        self.dataloader = dataloader

        # Initialize generator and discriminator
        self.generator = Generator(self.config.latent_dim, self.config.img_shape).to(self.device)
        self.discriminator = Discriminator(self.config.img_shape).to(self.device)

        # Optimizers
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=self.config.lr)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.config.lr)

        # Track loss per epoch
        self.g_losses = []
        self.d_losses = []

    def train(self):
        # ----------
        #  Training
        # ----------
        t0 = time.time()
        batches_done = 0
        for epoch in range(self.config.n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader): # self supervised --> no labels needed
                # Configure input
                real_imgs = torch.tensor(imgs, device=self.device, dtype=torch.float32)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], self.config.latent_dim)), device=self.device, dtype=torch.float32)

                # Generate a batch of images
                fake_imgs = self.generator(z).detach()

                # Adversarial loss
                loss_D = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(fake_imgs))

                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.config.clip_value, self.config.clip_value)

                # Train the generator every n_critic iterations
                if i % self.config.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    self.optimizer_G.zero_grad()

                    # Generate a batch of images
                    gen_imgs = self.generator(z)

                    # Adversarial loss
                    loss_G = -torch.mean(self.discriminator(gen_imgs))

                    loss_G.backward()
                    self.optimizer_G.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.config.n_epochs, batches_done % len(self.dataloader), len(self.dataloader), loss_D.item(), loss_G.item())
                    )

                # Save images per sample_interval
                # if batches_done % self.config.sample_interval == 0:
                #     save_image(gen_imgs.data[:25], f"wgan/{self.config.dataset}/%d.png" % batches_done, nrow=5, normalize=True)
                
                batches_done += 1

            # Save images per epoch
            save_image(gen_imgs.data[:25], f"wgan/{self.config.dataset}/%d.png" % epoch, nrow=5, normalize=True)
            # Save losses per epoch
            self.g_losses.append(loss_G.item())
            self.d_losses.append(loss_D.item())
            
        t1 = time.time()
        print(f"Training time for WGAN on {self.config.dataset} = {t1 - t0} sec")