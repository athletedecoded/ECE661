import os
import sys
import shutil
from types import SimpleNamespace

import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torchvision import datasets

import torch

from utils import plot_losses, load_config, build_dataloader, save_images

from gan.gan import GAN
from wgan.wgan import WGAN
from acgan.acgan import ACGAN
from wgangp.wgangp import WGANGP


def main():
    # Parse cli args
    if len(sys.argv) == 4:
        _, model, dataset, device = sys.argv[0], sys.argv[1], sys.argv[2], torch.device(sys.argv[3])
    else:
        _, model, dataset = sys.argv[0], sys.argv[1], sys.argv[2]
        device = None
    assert model in ["gan", "wgan", "acgan", "wgangp"]
    assert dataset in ["mnist", "cifar"]

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    config = SimpleNamespace(**load_config(f'{model}/config.yml'))
    config.dataset = dataset
    if dataset == "cifar":
        config.img_size = 32
        config.channels = 3
    elif dataset == "mnist":
        if model != "acgan":
            config.img_size = 28
        else:
            config.img_size = 32 # override as 32 for acgan
        config.channels = 1
    config.img_shape = (config.channels, config.img_size, config.img_size)

    # Construct dataloader
    dataloader = build_dataloader(dataset, config.img_size, config.batch_size)

    # Create fresh directory
    output_dir = f"{model}/{dataset}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(f"{output_dir}/gen_imgs")
    os.makedirs(f"{output_dir}/real_imgs")

    # Init model
    if model == "gan":
        mdl = GAN(config, dataloader, device)
    elif model == "wgan":
        mdl = WGAN(config, dataloader, device)
    elif model == "acgan":
        mdl = ACGAN(config, dataloader, device)
    elif model == "wgangp":
        mdl = WGANGP(config, dataloader, device)
    else:
        raise Exception("ERROR: Incorrect model useage must be one of gan, wgan, wgangp, acgan")

    # Run training
    mdl.train()

    # Plot losses
    plot_losses(output_dir, mdl.g_losses, mdl.d_losses, model, config.log_k_epoch)

    # Save images for FID
    save_images(mdl.gen_imgs, f"{model}/{dataset}/gen_imgs")
    save_images(mdl.real_imgs, f"{model}/{dataset}/real_imgs")

if __name__ == "__main__":
    main()