# ECE661
ECE661 Neural Networks Final Project

**Setup**
```
$ pip3 install virtualenv
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

**Useage**

* MODEL: 'gan', 'wgan', 'acgan', 'wgangp'
* DATASET: 'mnist', 'cifar'
* DEVICE: "cuda", "cpu" (Optional -- will auto select based on cuda availability)

```
$ python3 run.py <MODEL> <DATASET> <DEVICE>

# ex. python3 run.py acgan cifar
```

**FID Score**

```
# ex. python3 -m pytorch_fid wgan/cifar/real_imgs wgan_cifar/gen_images
```