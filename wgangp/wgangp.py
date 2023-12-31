import numpy as np
import time

from torchvision.utils import save_image
from utils import plot_losses, save_fid_images

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from utils import init_wts_normal


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
            nn.Linear(1024, int(np.prod(self.img_shape))),
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

class WGANGP():
    def __init__(self, config, dataloader, device):
        super(WGANGP, self).__init__()
        # Runtime configs
        self.config = config
        self.device = device
        self.dataloader = dataloader

        # Initialize generator and discriminator
        self.generator = Generator(self.config.latent_dim, self.config.img_shape).to(self.device)
        self.discriminator = Discriminator(self.config.img_shape).to(self.device)

        # Initialize weights
        self.generator.apply(init_wts_normal)
        self.discriminator.apply(init_wts_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2))

        # Track loss per epoch
        self.g_losses = []
        self.d_losses = []

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), device=self.device, dtype=torch.float32)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.full((real_samples.size(0), 1), 0.0, device=self.device, requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def train(self):
        t0 = time.time()
        # ----------
        #  Training
        # ----------
        batches_done = 0
        for epoch in range(self.config.n_epochs):
            # Save final image state
            self.fake_imgs = []
            self.real_imgs = []
            for i, (imgs, _) in enumerate(self.dataloader):

                # Configure input
                real_imgs = torch.tensor(imgs, device=self.device, dtype=torch.float32)

                # Log final epoch of images
                if epoch == self.config.n_epochs - 1:
                    self.real_imgs.append(real_imgs)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], self.config.latent_dim)), device=self.device, dtype=torch.float32)

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Real images
                real_validity = self.discriminator(real_imgs)
                # Fake images
                fake_validity = self.discriminator(gen_imgs)
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_imgs.data, gen_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.config.lambda_gp * gradient_penalty

                d_loss.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % self.config.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    gen_imgs = self.generator(z)

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(gen_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optimizer_G.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.config.n_epochs, i, len(self.dataloader), d_loss.item(), g_loss.item())
                    )

                    batches_done += self.config.n_critic

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
        print(f"Training time for WGANGP on {self.config.dataset} = {t1 - t0} sec")
