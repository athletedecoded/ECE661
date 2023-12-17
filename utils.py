import os
import yaml
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def init_wts_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def plot_losses(save_pth, g_losses, d_losses, model, k):
    if k == 1:
        epochs = range(0, len(g_losses))
    else:
        epochs = range(0, k*len(g_losses) - 1, k)
    plt.figure()
    plt.plot(epochs, g_losses, label='Generator')
    plt.plot(epochs, d_losses, label='Discriminator')
    plt.title(f'{model} Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.savefig(f'{save_pth}/losses.png')

def build_dataloader(dataset, img_size, channels, batch_size):
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
                [transforms.Resize(img_size), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5]*channels, [0.5]*channels)]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader

def save_fid_images(image_array, output_dir):
    for i, image in enumerate(image_array):
        img_path = os.path.join(output_dir, f'image_{i + 1}.png')
        save_image(image, img_path)