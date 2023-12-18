# Comparative Analysis of Generative Adversarial Networks (GANs)

**ECE661 Computer Engineering Machine Learning and Deep Neural Nets**

Team Members: Dhyay Bhatt & Kahlia Hogg

*Duke University, Fall 2023*

### Project Overview

Our project seeks to compare the performance of the following GAN architectures on MNIST and CIFAR.

* Original GAN (GAN) [[code]](./gan/gan.py) [[paper]](https://arxiv.org/abs/1406.2661)
* Wasserstein GAN (WGAN) [[code]](./wgan/wgan.py) [[paper]](https://arxiv.org/abs/1701.07875)
* Wasserstein GAN + Gradient Penalty (WGAN-GP) [[code]](./wgangp/wgangp.py) [[paper]](https://arxiv.org/abs/1704.00028)
* Auxiliary Classifier GAN (AC-GAN) [[code]](./acgan/acgan.py) [[paper]](https://arxiv.org/abs/1610.09585)


### Install

```
$ pip3 install virtualenv
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Useage

* MODEL: 'gan', 'wgan', 'acgan', 'wgangp'
* DATASET: 'mnist', 'cifar'
* DEVICE: "cuda", "cpu" (Optional -- will auto select based on cuda availability)

```
$ python3 run.py <MODEL> <DATASET> <DEVICE>

# ex. python3 run.py acgan cifar
```

### FID Analysis

```
# ex. python3 -m pytorch_fid wgan/cifar/real_imgs wgan/cifar/gen_imgs
```

### Results

**MNIST**

|   Model   | Num Epochs |  Runtime   |   FID   |
|:----------:|:----------:|:----------:|:-------:|
|    GAN     |    500     |  6545.27   | 105.37  |
|   WGAN     |    200     | 2293.0025  |  63.46  |
|  WGANGP    |    200     | 2661.8567  |  90.59  |
|   ACGAN*   |    200     |  7515.25   |  22.92  |


<hr>

**CIFAR**

|   Model   | Num Epochs |   Runtime   |   FID   |
|:----------:|:----------:|:-----------:|:-------:|
|    GAN     |    500     | 5990.0176   | 176.14  |
|   WGAN     |    200     |  2195.66    | 156.54  |
|  WGANGP    |    200     |  2499.04    | 304.61  |
|   ACGAN*   |    200     |  6036.49    | 133.45  |


*ACGAN models were run on Nvidia T4 GPU. All other models were run on Nvidia A10G GPU.