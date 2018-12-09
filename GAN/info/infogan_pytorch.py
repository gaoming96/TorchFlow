# -*- coding: utf-8 -*-
"""
Author: Ming Gao

Info GAN_PT.

Discriminator: dimension flow: 784->128->1, relu+sigmod
Generator: dimension flow: 16+10->256->784, relu+sigmod
Q:                                     784->128->10, relu+softmax.

Use xavier to init weights; use U(-1,1) to init z. (z:[batch,16]); 
use N(1,1) to init c. (c:[batch,10]).
c may control the width or incline.

mb_size=64, batch size.
mnist.train.next_batch(mb_size)[0].shape==(128, 784), range=[0,1]. (the reason why we use sigmod above)

(DLOSS,GLOSS) iter0: (1.1,3.2); iter5000: (0.2, 4); iter10000: (0.7,3).

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
    inputs = torch.cat([z, c], 1)
    h=F.linear(inputs,torch.transpose(Wzh, 0, 1),bzh)
    h = torch.relu(h)
    X=F.linear(h,torch.transpose(Whx, 0, 1),bhx)
    X = torch.sigmoid(X)
    return X


""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = torch.zeros(h_dim, requires_grad=True)

Why = xavier_init(size=[h_dim, 1])
bhy = torch.zeros(1, requires_grad=True)


def D(X):
    h=F.linear(X,torch.transpose(Wxh, 0, 1),bxh)
    h = torch.relu(h)
    y=F.linear(h,torch.transpose(Why, 0, 1),bhy)
    y = torch.sigmoid(y)
    return y


""" ====================== Q(c|X) ========================== """

Wqxh = xavier_init(size=[X_dim, h_dim])
bqxh = torch.zeros(h_dim, requires_grad=True)

Whc = xavier_init(size=[h_dim, 10])
bhc = torch.zeros(10, requires_grad=True)


def Q(X):
    h=F.linear(X,torch.transpose(Wqxh, 0, 1),bqxh)
    h = torch.relu(h)
    c=F.linear(h,torch.transpose(Whc, 0, 1),bhc)
    c = F.softmax(c,dim=1)
    return c


G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]
Q_params = [Wqxh, bqxh, Whc, bhc]


""" ===================== TRAINING ======================== """

G_solver = optim.Adam(G_params, lr=1e-3)
D_solver = optim.Adam(D_params, lr=1e-3)
Q_solver = optim.Adam(G_params + Q_params, lr=1e-3)

ones_label = torch.ones((mb_size, 1),requires_grad=False)
zeros_label = torch.zeros((mb_size, 1),requires_grad=False)

k = 0
for it in range(15000):
    # Sample data
    z = torch.randn((mb_size, Z_dim),requires_grad=False)
    X,_ = mnist.train.next_batch(mb_size)
    X = torch.from_numpy(X)
    c = np.random.multinomial(1, 10*[0.1], size=mb_size)
    c = torch.from_numpy(c.astype('float32'))

    # Dicriminator forward-loss-backward-update
    G_sample = G(z,c)
    D_real = D(X)
    D_fake = D(G_sample)

    D_loss_real = F.binary_cross_entropy(D_real, ones_label)
    D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake

    D_solver.zero_grad()
    D_loss.backward()
    D_solver.step()

    # Generator forward-loss-backward-update
    z = torch.randn((mb_size, Z_dim),requires_grad=False)
    c = np.random.multinomial(1, 10*[0.1], size=mb_size)
    c = torch.from_numpy(c.astype('float32'))
    G_sample = G(z,c)
    D_fake = D(G_sample)

    G_loss = F.binary_cross_entropy(D_fake, ones_label)

    G_solver.zero_grad()
    G_loss.backward()
    G_solver.step()
    
    # Q forward-loss-backward-update
    z = torch.randn((mb_size, Z_dim),requires_grad=False)
    c = np.random.multinomial(1, 10*[0.1], size=mb_size)
    c = torch.from_numpy(c.astype('float32'))
    G_sample = G(z, c)
    Q_c_given_x = Q(G_sample)

    Q_loss = torch.mean(-torch.sum(c * torch.log(Q_c_given_x + 1e-8), dim=1))
    
    Q_solver.zero_grad()
    Q_loss.backward()
    Q_solver.step()


    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.numpy(), G_loss.data.numpy()))
        
        
        #idx = np.random.randint(0, 10)
        idx=list(range(10))*6+[0]*4
        c = np.zeros([mb_size, 10])
        c[range(mb_size), idx] = 1
        c = torch.from_numpy(c.astype('float32'))
        samples = G(z, c).data.numpy()[:16]
        
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
