# -*- coding: utf-8 -*-
"""
Author: Ming Gao

Conditonal GAN_PT.

Discriminator: dimension flow: 784+10->128->1, relu+sigmod
Generator: dimension flow: 100+10->128->784+10, relu+sigmod

Use xavier to init weights; use U(-1,1) to init z.

mb_size=64, batch size.
mnist.train.next_batch(mb_size)[0].shape==(128, 784), range=[0,1]. (the reason why we use sigmod above)
mnist.train.next_batch(mb_size)[1].shape=(128, 10) (one-hot). 

We set x and y as the input. CGAN can both predict a figure and generate selected number's figure.

(DLOSS,GLOSS) iter0: (1.6,2); iter5000: (0.2, 7); iter10000: (0.8,2.5).

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
k = 0


#def xavier_init(size):
#    in_dim = size[0]
#    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
#    return torch.randn(size, requires_grad=True) * xavier_stddev

#def xavier_init(size):
#    in_dim = size[0]
#    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
#    return torch.tensor(torch.distributions.Normal(0,xavier_stddev).sample(),requires_grad=True)

# We cannot use the above to define weights. (problem occurs with * xavier_stddev 
# because we do an mal-operation). 

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return torch.autograd.Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)



""" ==================== GENERATOR ======================== """

Wzh = xavier_init(size=[Z_dim + y_dim, h_dim])
bzh = torch.zeros(h_dim, requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = torch.zeros(X_dim, requires_grad=True)


def G(z,y):
    inputs = torch.cat([z, y], 1)
    h=F.linear(inputs,torch.transpose(Wzh, 0, 1),bzh)
    h = torch.relu(h)
    X=F.linear(h,torch.transpose(Whx, 0, 1),bhx)
    X = torch.sigmoid(X)
    return X


""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[X_dim + y_dim, h_dim])
bxh = torch.zeros(h_dim, requires_grad=True)

Why = xavier_init(size=[h_dim, 1])
bhy = torch.zeros(1, requires_grad=True)


def D(X,y):
    inputs = torch.cat([X, y], 1)
    h=F.linear(inputs,torch.transpose(Wxh, 0, 1),bxh)
    h = torch.relu(h)
    y=F.linear(h,torch.transpose(Why, 0, 1),bhy)
    y = torch.sigmoid(y)
    return y


G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]


""" ===================== TRAINING ======================== """

G_solver = optim.Adam(G_params, lr=1e-3)
D_solver = optim.Adam(D_params, lr=1e-3)

ones_label = torch.ones((mb_size, 1),requires_grad=False)
zeros_label = torch.zeros((mb_size, 1),requires_grad=False)


for it in range(10000):
    # Sample data
    z = torch.randn((mb_size, Z_dim),requires_grad=False)
    X, y = mnist.train.next_batch(mb_size)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y.astype('float32'))


    # Dicriminator forward-loss-backward-update
    G_sample = G(z,y)
    D_real = D(X,y)
    D_fake = D(G_sample,y)

    D_loss_real = F.binary_cross_entropy(D_real, ones_label)
    D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake

    D_solver.zero_grad()
    D_loss.backward()
    D_solver.step()

    # Generator forward-loss-backward-update
    z = torch.randn((mb_size, Z_dim),requires_grad=False)
    G_sample = G(z,y)
    D_fake = D(G_sample,y)

    G_loss = F.binary_cross_entropy(D_fake, ones_label)

    G_solver.zero_grad()
    G_loss.backward()
    G_solver.step()


    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.numpy(), G_loss.data.numpy()))

        c = np.zeros(shape=[mb_size, y_dim], dtype='float32')
        c[:, 8] = 1.
        c = torch.from_numpy(c.astype('float32'))
        samples = G(z,c).data.numpy()[:16]

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

        plt.savefig('out/{}.png'.format(str(k).zfill(3)), bbox_inches='tight')
        k += 1
        plt.close(fig)