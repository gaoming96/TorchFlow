## Getting Started
First we clone this repo and install the required packages. We use MNIST dataset as the input. We may change some hyper paramters such as batch_size and traning_epoch to accelerate accuracy. The output figures is the subfolder: out/.

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

![](.././pics/vae_structure.png)

### Model structure of CVAE:
- Mean: dimension flow: 784+10->128->100, relu+linear
- Logvar: dimension flow: 784+10->128->100, relu+linear
- Q(z|X): sample z~N(mean,var). [batch, 100]
- P(X|z): dimension flow: 100+10->128->784, relu+sigmod
- LOSS=E[log P(X|z)]+KL(Q(z|X) || N(0,1)). First loss is ordinary cross entrophy
- We set x and y as the input. CVAE can both predict a figure and generate selected label's figure
