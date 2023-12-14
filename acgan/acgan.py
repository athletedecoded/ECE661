import time
import numpy as np

from torchvision.utils import save_image

from utils import init_wts_normal

import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_size, channels):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels, img_size, n_classes):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

class ACGAN():
    def __init__(self, config, dataloader, device):
        super(ACGAN, self).__init__()
        # Runtime configs
        self.config = config
        self.device = device
        self.dataloader = dataloader
        # Loss functions
        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()

        # Initialize generator and discriminator
        self.generator = Generator(self.config.n_classes, self.config.latent_dim, self.config.img_size, self.config.channels).to(device)
        self.discriminator = Discriminator(self.config.channels, self.config.img_size, self.config.n_classes).to(device)

        # Initialize weights
        self.generator.apply(init_wts_normal)
        self.discriminator.apply(init_wts_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2))

    def sample_image(self, n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, self.config.latent_dim)), device=self.device, dtype=torch.float32)
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = torch.tensor(labels, device=self.device, dtype=torch.long)
        gen_imgs = self.generator(z, labels)
        save_image(gen_imgs.data, f"acgan/{self.config.dataset}/%d.png" % batches_done, nrow=n_row, normalize=True)

    def train(self):
    # ----------
    #  Training
    # ----------
        t0 = time.time()
        for epoch in range(self.config.n_epochs):
            for i, (imgs, labels) in enumerate(self.dataloader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = torch.full((imgs.size(0), 1), 1.0, device=self.device, requires_grad=False)
                fake = torch.full((imgs.size(0), 1), 0.0, device=self.device, requires_grad=False)

                # Configure input
                real_imgs = torch.tensor(imgs, device=self.device, dtype=torch.float32)
                labels = torch.tensor(labels, device=self.device, dtype=torch.long)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = torch.tensor(np.random.normal(0, 1, (batch_size, self.config.latent_dim)), device=self.device, dtype=torch.float32)
                gen_labels = torch.tensor(np.random.randint(0, self.config.n_classes, batch_size), device=self.device, dtype=torch.long)

                # Generate a batch of images
                gen_imgs = self.generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = self.discriminator(gen_imgs)
                g_loss = 0.5 * (self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels))

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Loss for real images
                real_pred, real_aux = self.discriminator(real_imgs)
                d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = self.discriminator(gen_imgs.detach())
                d_fake_loss = (self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                d_loss.backward()
                self.optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (epoch, self.config.n_epochs, i, len(self.dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
                )
                batches_done = epoch * len(self.dataloader) + i

                # Save image per sample_interval
                # if batches_done % self.config.sample_interval == 0:
                #     self.sample_image(n_row=10, batches_done=batches_done)
            
            # Save per epoch
            self.sample_image(n_row=10, batches_done=epoch)   
        t1 = time.time()
        print(f"Training time for AC-GAN on {self.config.dataset} = {t1 - t0} sec")