import numpy as np
import time

from torchvision.utils import save_image
from utils import plot_losses, save_fid_images

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

        # Track loss per epoch
        self.g_losses = []
        self.d_losses = []

    # ----------
    #  Training
    # ----------
    def train(self):
        t0 = time.time()
        for epoch in range(self.config.n_epochs):
            # Log image state per log_k_epoch
            self.fake_imgs = []
            self.real_imgs = []
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
                
                # Collect kth epoch and final epoch images
                if (epoch % self.config.log_k_epoch == 0) or (epoch == self.config.n_epochs - 1):
                    self.real_imgs.append(real_imgs)
                    self.fake_imgs.append(gen_imgs)
            
            # logging at kth and final epochs
            if (epoch % self.config.log_k_epoch == 0) or (epoch == self.config.n_epochs - 1):
                # Log loss state
                self.g_losses.append(g_loss.item())
                self.d_losses.append(d_loss.item()) 
                # Plot losses
                plot_losses(f"{self.config.model}/{self.config.dataset}", self.g_losses, self.d_losses, self.config.model, self.config.log_k_epoch)
                # Save samples
                save_image(gen_imgs.data[:25], f"{self.config.model}/{self.config.dataset}/%d.png" % epoch, nrow=5, normalize=True)  
                save_image(real_imgs.data[:25], f"{self.config.model}/{self.config.dataset}/%d_real.png" % epoch, nrow=5, normalize=True)
                # Save 1k images for FID
                fid_gen_imgs = torch.cat(self.fake_imgs, dim=0)[:1000]
                fid_real_imgs = torch.cat(self.real_imgs, dim=0)[:1000]     
                save_fid_images(fid_gen_imgs, f"{self.config.model}/{self.config.dataset}/gen_imgs")
                save_fid_images(fid_real_imgs, f"{self.config.model}/{self.config.dataset}/real_imgs") 
                
        t1 = time.time()
        print(f"Training time for GAN on {self.config.dataset} = {t1 - t0} sec")