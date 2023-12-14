import torch
import matplotlib.pyplot as plt

def init_wts_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def plot_losses(save_pth, g_losses, d_losses):
    epochs = range(1, len(g_losses) + 1)
    plt.plot(epochs, g_losses, label='Generator')
    plt.plot(epochs, d_losses, label='Discriminator')
    plt.title('Generator vs Discriminator Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.savefig(f'{save_pth}/losses.png')
