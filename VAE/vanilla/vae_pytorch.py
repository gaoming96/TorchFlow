# -*- coding: utf-8 -*-
"""
Author: Ming Gao

Vanilla VAE_PT.

Mean: dimension flow: 784->128->100, relu+linear
Logvar: dimension flow: 784->128->100, relu+linear
Sample z~N(mean,var). [batch, 100]

P(X|z): dimension flow: 100->128->784, relu+sigmod

Use xavier to init weights.

mb_size=64, batch size.
mnist.train.next_batch(mb_size)[0].shape==(128, 784), range=[0,1].

LOSS=E[log P(X|z)]+KL(Q(z|X) || N(0,1)). First one is ordinary cross entrophy (we regard X as y).

LOSS iter0: 760; iter5000: 119; iter10000: 108.

"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return torch.autograd.Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


# =============================== Q(z|X) ======================================

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = torch.zeros(h_dim, requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = torch.zeros(Z_dim, requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = torch.zeros(Z_dim, requires_grad=True)


def Q(X):
    h=F.linear(X,torch.transpose(Wxh, 0, 1),bxh)
    h = torch.relu(h)
    z_mu=F.linear(h,torch.transpose(Whz_mu, 0, 1),bhz_mu)
    z_var=F.linear(h,torch.transpose(Whz_var, 0, 1),bhz_var)
    
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = torch.randn((mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = torch.zeros(h_dim, requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = torch.zeros(X_dim, requires_grad=True)


def P(z):
    h=F.linear(z,torch.transpose(Wzh, 0, 1),bzh)
    h = torch.relu(h)
    X=F.linear(h,torch.transpose(Whx, 0, 1),bhx)
    X=torch.sigmoid(X)

    return X


# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
          Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)

for it in range(10000):
    X, _ = mnist.train.next_batch(mb_size)
    X = torch.from_numpy(X)

    # Forward
    z_mu, z_var = Q(X)
    z = sample_z(z_mu, z_var)
    X_sample = P(z)

    # Loss
    recon_loss = F.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + kl_loss

    solver.zero_grad()
    # Backward
    loss.backward()

    # Update
    solver.step()


    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))

        samples = P(z).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)
