## Getting Started
First we clone this repo and install the required packages. We use MNIST dataset as the input. We may change some hyper paramters such as batch_size and traning_epoch to accelerate accuracy. The output figures is the subfolder: out/.

## Generative Adversarial Nets (GAN)
1. [Vanilla GAN](https://arxiv.org/abs/1406.2661)
2. [Conditional GAN](https://arxiv.org/abs/1411.1784)
3. [InfoGAN](https://arxiv.org/abs/1606.03657)
4. [Cycle GAN](https://arxiv.org/pdf/1703.10593.pdf)

GAN trains a discriminator and generator, which is adversarial.

The codes are based on original papers and wiseodd/generative-models, however, I make some improvements:
1. I try to make both Pytorch and Tensorflow's code similarly to each other.
2. Both codes are as simple and concise as possible (don't use argparse or some fancy utils).
3. Both codes are updated to the latest version (TF: API r1.12, PT: Version 1.0 (`Variable` is deprecated)).
4. Since it is used for self learning, I don't use higer API (such as keras, eager, layer) and I specify weights explicity, so it is more understandable. However, because it is too complicated in CycleGAN, I use `torch.nn` and `tf.nn` to build layers.

### Model structure of GAN:
- Discriminator: dimension flow: 784->128->1, relu+sigmod
- Generator: dimension flow: 100->128->784, relu+sigmod
- Use xavier to init weights; use U(-1,1) to init z
- DLOSS=`binary_cross_entropy(D_real, ones_label)+binary_cross_entropy(D_fake, zeros_label)`
- GLOSS=`binary_cross_entropy(D_fake, ones_label)`

### Model structure of CGAN:
- Discriminator: dimension flow: 784+10->128->1, relu+sigmod
- Generator: dimension flow: 100+10->128->784+10, relu+sigmod
- Use xavier to init weights; use U(-1,1) to init z
- We set x and y as the input. CVAE can both predict a figure and generate selected label's figure

### Model structure of InfoGAN:
- Discriminator: dimension flow: 784->128->1, relu+sigmod
- Generator: dimension flow: 16+10->256->784, relu+sigmod
- Q:                                     784->128->10, relu+softmax
- Use xavier to init weights; use U(-1,1) to init z (z:[batch,16]); use N(1,1) to init c (c:[batch,10])
- c may control the width or incline
- QLOSS=E(P(Q(G_sample)|c))

Model structure of GANs: (left is GAN, left+red is CGAN, right is InfoGAN)

![](./pics/gan_structure.png)

### Model structure of CycleGAN:
In CycleGAN, we have two datasets and we don't need to sample noise z.

![](./pics/cycle_gan_structure.png)

Exemplar results on testset: horse -> zebra

![](./pics/horse2zebra.gif)
![](./pics/horse2zebra1.gif)
