import yaml
import os
import sys
from types import SimpleNamespace

import torchvision.transforms as transforms
from torchvision import datasets

import torch

from gan.gan import GAN
from wgan.wgan import WGAN
from acgan.acgan import ACGAN
from wgangp.wgangp import WGANGP

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def main():
    # Parse cli args
    _, model = sys.argv[0], sys.argv[1]
    assert model in ["gan", "wgan", "acgan", "wgangp"]
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load config
    config = SimpleNamespace(**load_config(f'{model}/config.yml'))
    config.img_shape = (config.channels, config.img_size, config.img_size)
    # Configure data loader
    os.makedirs("./data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(config.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=config.batch_size,
    shuffle=True,
    )
    # Init model
    if model == "gan":
        os.makedirs("gan/images", exist_ok=True)
        mdl = GAN(config, dataloader, device)
    elif model == "wgan":
        os.makedirs("wgan/images", exist_ok=True)
        mdl = WGAN(config, dataloader, device)
    elif model == "acgan":
        os.makedirs("acgan/images", exist_ok=True)
        mdl = ACGAN(config, dataloader, device)
    elif model == "wgangp":
        os.makedirs("wgangp/images", exist_ok=True)
        mdl = WGANGP(config, dataloader, device)
    else:
        raise Exception("ERROR: Incorrect model useage must be one of gan, wgan, wgangp, acgan")
    # Run training
    mdl.train()

if __name__ == "__main__":
    main()