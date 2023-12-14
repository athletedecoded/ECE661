import yaml
import os
import sys
import shutil
from types import SimpleNamespace

import torchvision.transforms as transforms
from torchvision import datasets

import torch

from utils import plot_losses

from gan.gan import GAN
from wgan.wgan import WGAN
from acgan.acgan import ACGAN
from wgangp.wgangp import WGANGP

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def build_dataloader(dataset, img_size, batch_size):
    os.makedirs(f"./data/{dataset}", exist_ok=True)
    if dataset == "mnist":
        ds = datasets.MNIST
    elif dataset == "cifar":
        ds = datasets.CIFAR10
    else:
        raise Exception("ERROR: Dataset must be one of mnist, cifar")

    dataloader = torch.utils.data.DataLoader(
        ds(
            f"./data/{dataset}",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader

def main():
    # Parse cli args
    _, model, dataset = sys.argv[0], sys.argv[1], sys.argv[2]
    assert model in ["gan", "wgan", "acgan", "wgangp"]
    assert dataset in ["mnist", "cifar"]

    # Set device
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
    os.makedirs(f"{model}/{dataset}")

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
    plot_losses(output_dir, mdl.g_losses, mdl.d_losses)

if __name__ == "__main__":
    main()