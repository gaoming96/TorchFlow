# TorchFlow
Some implementations using Pytorch and Tensorflow.

## Variational Autoencoder (VAE)
1. [Vanilla VAE](https://arxiv.org/abs/1312.6114)
2. [Conditional VAE](https://arxiv.org/abs/1406.5298)

VAE is the simplest Generative Model. It learns `mean` and `log_sd` of the latent variable. After the model is built, we may generate samples from a Normal distribution. The Conditional VAE has labels as the input and we can generate samples fixing the specific label.

The codes are based on original papers and wiseodd/generative-models, however, I make some improvements:
1. I try to make both Pytorch and Tensorflow's code similarly to each other.
2. Both codes are as simple and concise as possible (don't use argparse or some fancy utils).
3. Both codes are updated to the latest version (TF: API r1.12, PT: Version 1.0 (`Variable` is deprecated)).
4. Since it is used for self learning, I don't use higer API (such as keras, eager, layer) and I specify weights explicity, so it is more understandable.

### Model structure of VAE:
- Mean: dimension flow: 784->128->100, relu+linear
- Logvar: dimension flow: 784->128->100, relu+linear
- Q(z|X): sample z~N(mean,var). [batch, 100]
- P(X|z): dimension flow: 100->128->784, relu+sigmod
- LOSS=E[log P(X|z)]+KL(Q(z|X) || N(0,1)). First loss is ordinary cross entrophy

Model structure of VAE:

![](./pics/vae_structure.png)

### Model structure of CVAE:
- Mean: dimension flow: 784+10->128->100, relu+linear
- Logvar: dimension flow: 784+10->128->100, relu+linear
- Q(z|X): sample z~N(mean,var). [batch, 100]
- P(X|z): dimension flow: 100+10->128->784, relu+sigmod
- LOSS=E[log P(X|z)]+KL(Q(z|X) || N(0,1)). First loss is ordinary cross entrophy
- We set x and y as the input. CVAE can both predict a figure and generate selected label's figure


## Generative Adversarial Nets (GAN)
1. [Vanilla GAN](https://arxiv.org/abs/1406.2661)
2. [Conditional GAN](https://arxiv.org/abs/1411.1784)
3. [InfoGAN](https://arxiv.org/abs/1606.03657)
4. [Cycle GAN](https://arxiv.org/pdf/1703.10593.pdf)

GAN trains a discriminator and generator, which is adversarial. Generator G(z) tries to generate from noise z to the same distribution of X, while discriminator (\in [0,1]) tries to discriminate them.

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

Since in GAN, there is no restriction in z, which may hard for us to use the information. In infoGAN, we seperate z into random z and latent c. c can be categorical (Mulcat(10) to stand for labels 0-9) or continuous (Normal to stand for incline or width).

In LOSS, we add a Mutual Information regularization term: I(c || G(c,z)). Because c and G(c,z) should have high correlation.

However, because this regularization need posterior P(c|X), which is hard to get, we use Q(c,x) to approximate.

- Discriminator: dimension flow: 784->128->1, relu+sigmod
- Generator: dimension flow: 16+10->256->784, relu+sigmod
- Q:--------------------------------------------784->128->10, relu+softmax
- Use xavier to init weights; use U(-1,1) to init z (z:[batch,16]); use N(1,1) to init c (c:[batch,10])
- c may control the width or incline; Mulcat(10) of c may stand for labels
- QLOSS=E(P(Q(G_sample)|c))

Model structure of GANs: (left is GAN, left+red is CGAN, right is InfoGAN)

![](./pics/gan_structure.png)

### Model structure of CycleGAN:
In CycleGAN, we have two datasets and we don't need to sample noise z.

![](./pics/cycle_gan_structure.png)
![](./pics/cycle_gan_structure1.png)

Exemplar results on testset: horse -> zebra

![](./pics/horse2zebra.gif)
![](./pics/horse2zebra1.gif)
