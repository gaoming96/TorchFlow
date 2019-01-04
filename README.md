# TorchFlow
Some implementations using Pytorch and Tensorflow.

## API

### Tensorflow
In VAE and GAN (except CycleGAN), we use `tf.Variable` to construct weights explicitly. We only use `tf.matmul` to define layers, `tf.nn.sigmoid`, `tf.nn.relu` or `tf.nn.softmax` to make activations. For CNN, we use `tf.nn.conv2d`. 

The advantage of just using tf base functions is that when updating a subset of weights, we can use
```python
w1=tf.Variable()
w2=tf.Variable()
theta_some = [w1, w2]

solver = tf.train.AdamOptimizer().minimize(loss, var_list=theta_some)
```

In comparison, in CycleGAN and RNN, we don't define weights ourselves. **We use `tf.layers.dense`, `tf.layers.conv2d` for CNN and 
`tf.nn.rnn_cell` \& `tf.nn.static_rnn` for RNN.**

If we want to update a subset of weights, we need to state layers' names.

### Pytorch
In VAE and GAN (except CycleGAN), we use `torch.zeros` or `torch.randn` to construct weights explicitly. We only use `torch.nn.functional.linear` to define layers, `torch.relu`, `torch.sigmoid` or `torch.softmax` to make activations.

For loss, we use `torch.nn.CrossEntropyLoss` (using `logits`) or `F.binary_cross_entropy` (using `sigmoid(logits)`).

In comparison, in CycleGAN, RNN and CNN, we don't define weights ourselves. **We use `torch.nn.Conv2d` \& `torch.nn.MaxPool2d` for CNN, 
`nn.Linear` for MLP, `nn.RNN` for RNN. Here, we can use several Class inherit from `nn.module` and update weights easily.**
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x
        
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for i, data in enumerate(trainloader):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

## CNN
1. [cifar10](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) (PT)
2. [spatial transformer](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html) (PT)
3. [transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) (PT)

### cifar10:
In this pytorch official tutorial, we emphasis on `DataLoader`, CNN structure, `torch.no_grad()` in test step and GPU usage.

key code:
```python
# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].
transform = transforms.Compose( [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# set `num_workers=0` for windows user. Or error: BrokenPipeError: [Errno 32] Broken pipe. See: https://github.com/pytorch/pytorch/issues/2341
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

class Net(nn.Module):

net = Net()

# test step
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        #...
```

### Spacial Transformer Network (STN):

STN allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric
invariance of the model.

For example, it can crop a region of interest, scale and correct the orientation of an image. It can be a useful mechanism because CNNs
are not invariant to rotation and scale and more general affine transformations.

So, if images scale or rotate a lot, CNN can't learn well. Use STN. STN can not only add accuracy but also use as generative model: we can show figures (sampler) after scaled and rotated.

One of the best things about STN is the ability to simply plug it into any existing CNN with very little modification.

![](./pics/stn_structure.jpg)

**We don't know the correct ground truth of the rotation and scaling, we use a NN (localizaiton) to learn and it works well!**

#### Flow:
1. input: x: [64, 1, 28, 28] (figure), label: [64]. batch_size=64.
2. Spatial transformer localization-network: `xs=localization(x)`: [64, 1, 28, 28] -> [64, 10, 3, 3]. (Conv2d+MaxPool2d+ReLU)*2.
3. Regressor for the 3 * 2 affine matrix: `theta=fc_loc(xs.view(-1, 10*3*3)).view(-1, 2, 3)`.
    fc_loc: [64, 10\*3\*3] -> [64, 32\*3\*2]. (Linear+ReLU+Linear).
4.  grid generator \& sampler: `x = F.grid_sample(x, F.affine_grid(theta, x.size()) )`. [64, 32\*3\*2] -> [64, 1, 28, 28].
5. ordinary CNN: `x=CNN(x)`, loss and train step.

We look closer to the structure.

In localization, we have input: [64, 1, 28, 28] (consider [28,28] instead). Each point in figure is a coordinate (x,y)' (1<=x<=28).
We use affine transformation: (xs,ys)'=[theta; 2\*3 matrix]\*(xt,yt,1)' to represent all the transformation. xt is target (output) while xs is sourse (input).

Now, we learn theta from a NN and we can get coordinate (xt,yt). However, this coordinate is not integer. So we use kernel to represent distance of the non-integer coordinate and all the interger grid coordinate.

![equation](https://latex.codecogs.com/gif.latex?V%20_%20%7B%20i%20%7D%20%5E%20%7B%20c%20%7D%20%3D%20%5Csum%20_%20%7B%20n%20%7D%20%5E%20%7B%20H%20%7D%20%5Csum%20_%20%7B%20m%20%7D%20%5E%20%7B%20W%20%7D%20U%20_%20%7B%20n%20m%20%7D%20%5E%20%7B%20c%20%7D%20%5Cmax%20%5Cleft%28%200%2C1%20-%20%5Cleft%7C%20x%20_%20%7B%20i%20%7D%20%5E%20%7B%20s%20%7D%20-%20m%20%5Cright%7C%20%5Cright%29%20%5Cmax%20%5Cleft%28%200%2C1%20-%20%5Cleft%7C%20y%20_%20%7B%20i%20%7D%20%5E%20%7B%20s%20%7D%20-%20n%20%5Cright%7C%20%5Cright%29)

Here, Umnc is the input at location (m,n) in channel c, Vic is the output at location (xit,yit).

One note about Class `nn.module`. If we use `F.dropout` or batchnorm, which acts differently in train and test step, we can do this:
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ...
    def forward(self, x):
        x = F.dropout(x, training=self.training)
        ...
model = Net()

#train
model.train()
#update

#test
with torch.no_grad():
    model.eval()
    #test
```
### Transfer learning:
In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.

In this tutorial, we focus on:

First, learning rate decay in several round.
```python
from torch.optim import lr_scheduler

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# train step
for epoch in range(num_epochs):
    exp_lr_scheduler.step()
```
Second, fine tune the model. If we want to fix weights of the upper layers, we can set these weights to `requires_grad=F`.
```python
from torchvision import models

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

model_ft.fc
# Linear(in_features=512, out_features=1000, bias=True)

# change the last layer to outfeatures=2 to realize fine tune
model_ft.fc = nn.Linear(num_ftrs, 2)
```

Third, methods and attributes of class `nn.module`.
``` python
model_ft = models.resnet18(pretrained=True)
# get specific layer's structure
model_ft.fc
# get this layer's parameter
model_ft.fc.in_features
# get this layer's current weigths
model_ft.fc.weight
model_ft.fc.bias
# get all weights and bias
list(model_ft.parameters())
model_ft.state_dict()
# get training mode or eval mode
model_ft.training
# True
```

## Variational Autoencoder (VAE)
1. [Vanilla VAE](https://arxiv.org/abs/1312.6114)
2. [Conditional VAE](https://arxiv.org/abs/1406.5298)

VAE is the simplest Generative Model. It learns `mean` and `log_sd` of the latent variable. After the model is built, we may generate samples from a Normal distribution. The Conditional VAE has labels as the input and we can generate samples fixing the specific label.

The codes are based on original papers and [wiseodd/generative-models](https://github.com/wiseodd/generative-models), however, I make some improvements:
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
1. [Vanilla GAN]
2. [Conditional GAN](https://arxiv.org/abs/1411.1784)
3. [InfoGAN]
4. [Cycle GAN]
5. [DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) (PT)
6. [BEGAN]

GAN trains a discriminator and generator, which is adversarial. Generator G(z) tries to generate from noise z to the same distribution of X, while discriminator (\in [0,1]) tries to discriminate them.

The codes are based on original papers and [wiseodd/generative-models](https://github.com/wiseodd/generative-models).

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

### Model structure of DCGAN:
DCGAN explicitly uses convolutional and convolutional-transpose layers in the discriminator and generator, respectively.

The discriminator is made up of strided convolution layers, batch norm layers, and LeakyReLU activations. The generator is comprised of convolutional-transpose layers, batch norm layers, and ReLU activations.

1. Celeb-A Faces dataset. See [Pytorch example](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) for more details.
2. Weight Initialization ~ N(0,0.2).
3. Use strided convolution rather than pooling to downsample because it lets the network learn its own pooling function.
4. Batch norm and leaky relu functions promote healthy gradient flow which is critical.
5. Both are Adam optimizers with learning rate 0.0002 and Beta1 = 0.5.
6. First generate a fixed batch of latent vectors that are drawn from a Gaussian distribution. Then in every training step, periodically input this fixed_noise to calculate loss.

It is a great code for CNN sequential and GPU device.

```python
# ConvTranspose2d
nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

# Conv2d
nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
```
### Boundary Equilibrium GAN
Disadvantage of GAN:

1. Hard to train, we need tricks (SRELU, batch norm, batch discrimination). Correct hyper-parameter selection is critical
2. Controlling the image diversity of the generated samples is difficult. 
3. Balancing the convergence of the discriminator and of the generator is a challenge: frequently the discriminator wins
too easily at the beginning of training
4. mode collapse, a failure mode in which just one image is learned (see playground/ProGAN for details)

BEGAN:

1. A GAN with a simple yet robust architecture, standard training procedure with fast and stable convergence (no tricks)
2. An equilibrium concept that balances the power of the discriminator against the generator (no too strong discriminator, equivalent of D&G)
3. A new way to control the trade-off between image diversity and visual quality (fantastic)
4. An approximate measure of convergence (similar to **Wasserstein GAN (WGAN)**)

Key idea: **matching the distribution of the errors instead of matching the distribution of the samples directly**.


### Model structure
![](./pics/began_structure.png)

Discriminator: Encoder+Decoder (output is a figure, not a number). Generator: the same in GAN. 

In Decoder step, upsampling is done by nearest neighbor (easy). G uses the same architecture (though not the same weights) as the
discriminator decoder. We made this choice only for simplicity. 

In the gray block, we can add a trick (skip connections) for more sharpness.

### LOSS
**we match the distribution of the errors instead of matching the distribution of the samples directly**.

First, we define a Loss function which measures the input fig and fig after Discriminator.

![](./pics/began_loss1.jpg)

x: real sample. L(x) is the loss. L(x) has its distribution (distribution of the errors). \mu1: Distribution of L(x).

z: N dim, z~U[-1,1]^N. \mu2: Distri of L(G(z)).

We then use Wasserstein distance of two distributions. Since it is too complex, we try to get lower bound: we compute a lower bound to the Wasserstein distance between the auto-encoder loss distributions of real and generated samples.

Using Jensen’s inequality, we can derive a lower bound **W(\mu1,\mu2)>=m2-m1**, m1=E(\mu1) [because m2>m1]

Discriminator is good == W(\mu1,\mu2) is small == min(m2-m1).

Generator is the opposite of D.

Now, we try to solve the problem of too strong Discri (D overwhelm G). We introduce \gamma as the diversity ratio.

\gamma==m2/m1, thus if \gamma lower, m2 lower, Discri greater, thus ...

![](./pics/began_loss2.jpg)

The right fig is the final LOSS to update.

```python
# update D
# real samples
D_real = self.discriminator(input)
D_loss_real = torch.mean(torch.abs(D_real - input))
# fake samples
X_b_fake = self.generator(z)
D_fake = self.discriminator(X_b_fake.detach())
D_loss_fake = torch.mean(torch.abs(D_fake - X_b_fake))
D_loss = D_loss_real - kt * D_loss_fake
D_loss.backward()
optimizer_D.step()

# update G
X_b_fake = self.generator(z)
D_fake = self.discriminator(X_b_fake)
G_loss = torch.mean(torch.abs(D_fake - X_b_fake))
G_loss.backward()
optimizer_G.step()

# update kt
kt = kt + lambda_k * (gamma * D_loss_real - G_loss)
kt = float(kt.cpu().data.numpy())
kt = min(1., max(0., kt))
```


## Recurrent Neural Network (RNN)
1. [Classifying names from languages](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) (PT)
2. [Generating names from languages](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html) (PT)
3. [Predicting hand-written number] (TF)
4. [Generating shakespeares play] (TF)

RNN trains a hidden state (in LSTM trains several gates and cell state) and in each sequence (time step), we use both input and current hidden state to compute the next state. After that, we use a linear network to convey hidden state into output.

**The difference between RNN and MLP is that it is dependent between sequence (time step).**

In the first and third examples, we see that RNN can also make **prediction**. We can predict a category after reading in all the letters of a name, and use the last time step (sequence) output to calculate cross-entrophy loss.

We need to sum the loss of each sequence in example 2 & 4, which is more commplicated and different from above examples.

The first two models are based on [Pytorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html), however, I make some improvements:
1. I try to make both Pytorch and Tensorflow's code similarly to each other.
2. Both codes are as simple and concise as possible (don't use argparse or some fancy utils).
3. Both codes are updated to the latest version (TF: API r1.12, PT: Version 1.0 (`Variable` is deprecated)).
4. In original Pytorch tutorial, the author defines RNN himself. However, I use `torch.nn.RNN`, which is more understandable.

### Model structures
Following figures are model structures of RNN, LSTM and Bidirectional-RNN.
![](./pics/rnn_structure.jpg)
![](./pics/rnn_structure.png)
![](./pics/lstm_structure.jpg)
![](./pics/bi_rnn_structure.png)

### Classifying names from languages (PT)
Given a name, we can predict the language used:
```python
$ python predict.py Schmidhuber
    (-0.19) German
    (-2.48) Czech
    (-2.68) Dutch
```
#### Flow:
1. Dataset: included in the ``data/names`` directory are 18 text files named as
"[Language].txt". Each file contains a bunch of names, one name per
line, mostly romanized (but we still need to convert from Unicode to
ASCII).
2. Input: In each round, one word (eg: `Hinton`), one output (`Scottish`).
3. `Hinton`:[seq=6, batch=1, input_dim=57] -> hidden:[seq=6, batch=1, hid_dim=128].
Then we use hidden[-1,:,:] to do linear network -> output:[1,18].
Finally, we use output and `Scottish`[1,18] (one hot) to compute loss.

**Note that in each time step, we feed in Hinton[i,:,:] to get hidden [i,:,:].**

**Note that hidden[-1,:,:] is the last time step, which can be regard as information we learnt from all the sequence (6 time step).**

We use batch=1, input_dim=57 (totally 57 characters in vocabulary dictionary),
hidden_dim=128, output_dim=18 (totally 18 languages which we want to classify).
eg: for word `Hinton`, seq=6 (6 characters in word `Hinton`).

We see that RNN can also make **prediction**. We can predict a category after reading in all the letters of a name, and use the last time step (sequence) output to calculate cross-entrophy loss.

Key codes:
```python
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.layer_dim, nonlinearity='relu')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[-1,:, :]) 
        return out
        
rnn = RNN(n_letters, n_hidden, 1,n_categories)
optimizer=optim.SGD(rnn.parameters(),lr=0.005)
criterion = nn.CrossEntropyLoss()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    optimizer.zero_grad()
    output = rnn(line_tensor)
    # line_tensor shape:(seq_len, batch, input_size); output shape:  (seq_len, batch, output_size)
    
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
``` 
### Generating names from languages (PT)
Generate names. [failed, I don't know why... Maybe because I forget to concate language with input, but if I only use Chinese language, it still not learnt]

#### Flow:
1. Dataset: included in the ``data/names`` directory are 18 text files named as
"[Language].txt". Each file contains a bunch of names, one name per
line, mostly romanized (but we still need to convert from Unicode to
ASCII).
2. Input: In each round, one word (eg: `Hinton`), one output (`intonEOS`). `EOS` means the end of word.
3. `Hinton`:[seq=6, batch=1, input_dim=59] -> hidden:[seq=6, batch=1, hid_dim=128].
4. Then we use hidden[i,:,:] to do linear network -> logit:[seq=6, batch=1, input_dim=59] (updating hidden each time step).
5. Finally, we use logit and output:[seq=6, batch=1, input_dim=59] to compute loss (sum of loss of each time step).
6. For generating step, given a language and a beginning letter `a` for example, put `a`:[1,1,59] in the model to get logit:[1,1,59] to get second letter. Repeat.

We use batch=1, input_dim=59 (totally 59 characters in vocabulary dictionary), hidden_dim=128.

### Predicting hand-written number (TF)

We set batch_size=64. Since a figure is 28\*28, we set seq=28 (each row of a figure) and input_dim=28. In this example, seq_length is fixed while in the above, seq depends on words.

#### Flow:
1. Input: In each round, 64 figures and their numbers.
2. [batch=64,seq=28, input_dim=28]-> (`tf.unstack`) 28@[batch=64,input_dim=28] -> hidden: 28@[batch=64, hid_dim=128].
Then we use hidden[-1] to do linear network -> output:[64,10].
Finally, we use output and number (one hot) to compute loss.

**Note that in each time step, we feed in input[i] to get hidden[i].**

**Note that hidden[-1] is the last time step, which can be regard as information we learnt from all the sequence (28 time steps).**

Key codes:
```python
tf.reset_default_graph()
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

X_list = tf.unstack(X, timesteps, 1)

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)
outputs, _ = tf.nn.static_rnn(lstm_cell, X_list, dtype=tf.float32)

logits = tf.layers.dense(outputs[-1], units=num_classes, activation=None,
                         kernel_initializer=tf.random_normal_initializer(seed=0))

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        _,loss=sess.run([train_op,loss], feed_dict={X: batch_x, Y: batch_y})
```

### Generating shakespeares play (TF)

The dataset is a period of shakespeares novel. We set batch=1, seq=25 (25 letters). We generate new chars after given the beginning of words.

#### Flow:
1. Input: (1,25,65). Totally 65 different chars in this novel. Output: (1,25,65), which is input moving forward a letter. eg: `Gaomin` is the input, then `aoming` is the output.
2. `tf.unstack` to 25@[1,65], then we RNN -> hidden: 25@[1,100].
3. In each hidden[i], we define linear network (reuse weight for all time step), to get logit: 25@[1,65].
4. We compute loss of logit: 25@[1,65] and output: 25@[1,65] (sum of all time step).
5. In train step, we use the current hidden state as the input hidden state in each epic (one epic: num of rounds that train all chars once), and then clear to 0 in next epic.
6. For test step, we first give 25 letters. Then each step, we choose the highest prob of the 26th letter. After that, we abandon the first letter and choose 2-26th letters as the input and repeat. We use current hidden state and reuse hidden-output as hidden-input.

**Note: here seq=25 is fixed. But we can set `seq` to be a `placeholder` to make it changeable in test part.**

## Word Embeddings

In RNN, we introduce the char-level input (every character is encoded into a vector). Now, each word is encoded into a number. We can turn this number into a vector by one-hot encoding, but the vocabulary size is too big and one-hot doesn't have Dristributed representation (mathematical vector property). Hence, we use NN to learn the latent word vector itself (embedding).

1. [NGram](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py) (PT)
2. [Skip-gram with negative sample] (TF)
3. [tag_word](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py) (PT)
4. [seq2seq translation](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py) (PT)
5. [chatbot](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html) (PT)

We introduce several similar word2vec methods: NGram, CBOW and Skip-Gram.

Also, since vocabulary size is too big, using ordinary softmax (MLP) costs time. We introduce hierarchy tree and negative sampling to solve the problem.

In the third tutorial, we combine word embedding with LSTM model. Always, we use word embdding as the first layer.

### NGram (PT)
We introduce several similar word2vec methods: NGram, CBOW and Skip-Gram.

1. the text: "the brown quick fox jumped ..."
2. NGram with context size=2: ([the, brown], quick), ([brown, quick], fox), ...
3. CBOW with window size=1: ([the, quick], brown), ([brown, fox], quick), ([quick, jumped], fox), ... Learn word from context
4. Skip-Gram with window size=1: (brown, the), (brown, quick), (quick, brown), (quick, fox), ... Learn context from word
5. Turn all the words into a number \in [0, voc_size-1]

We define  `embedding_dim=10`, `batch_size=1`, `context_size=2`. There are totally 97 different words.

1. Input: size=2, eg: [0,2] (represents [the,quick]). Output: size=1.
2. Dimension flow: [2] -> (`nn.Embedding(97,10)`) [2,10] -> (`view((1, -1))`) [1,20] -> [1, 97] (linear+relu)*2.
3. Ordinary Crossentrophy Loss.

Here, `nn.Embedding(97,10)` has learnable parameter size=[97,10] to represent the embedding: from 97 to 10. Since input contex=2 (2 words), we choose these 2 words from 97 words, hence we get [2,10]. Actually, we can regard [2] as [2,97] if we onehot it. Thus, embedding just embed [2,97] into [2,10].

Key code for embedding:

```python
embedding = nn.Embedding(10, 3)
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# torch.Size([2, 4])
embedding(input).size()
# torch.Size([2, 4, 3])
```

After the model is learned, we can add following words. Also, the embedding matrix [97,10] learns some semantic meanings.

### Skip-gram with negative sample (TF)
Since vocabulary size is too big, using ordinary softmax (MLP) costs time. We introduce hierarchy tree and negative sampling to solve the problem.

Here, we use Skip-gram with size=1. Embed_size=2 (for better visualization). batch_size=20. num_sample=15 (number of negative examples). There are totally voc_size=35 different words.
Flow:

1. Input: [batch], output: [batch,1]. eg: x=[2,10,0], y=[[1],[11],[30]] if batch=3.
2. Compute NCE loss and update.

embeddings:[35,2], embed:[batch,2] (the same as PT before). NCE loss is binary logistic regression of 1 true and 15 false obervations.

Key code:

```python
x = tf.placeholder(tf.int32, shape=[batch_size])
y = tf.placeholder(tf.int32, shape=[batch_size, 1])

embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, x) # lookup table

# Construct the variables for the NCE loss
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size],-1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, y, embed, num_sampled, voc_size))

train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(100):
        _, loss_val = sess.run([train_op, loss], feed_dict={x: batch_inputs, y: batch_labels})
    trained_embeddings = embeddings.eval()

# every word is a point in the embed figure
plot(trained_embeddings) 
```

### tag_word (PT)

Each sample of input is a sentence and its words' tags. eg: `("Everybody read that book".split(), ["NN", "V", "DET", "NN"])`.
We turn into numbers: `(tensor([5, 6, 7, 8]), tensor([1, 2, 0, 1]))`.

We set EMBEDDING_DIM = 6, HIDDEN_DIM = 16, vocab_size=9 (total 9 words), tagset_size=3 (total 3 tags), 

Dim flow:

1. input: [4] (if there are 4 words in this sentence) -> (`nn.Embedding(9,6)`) [4,6] -> (`view`) [4,1,6] (batch=1,seq=4)
2. LSTM layer: -> (`nn.LSTM(6,16)`) [4,1,16] -> (`view(4,-1)`) [4,16] -> [4,3]
3. Ordinary loss, with desired output: [4] -> (onehot) [4,3]
4. Hidden state is updating each round, and clear to 0 each epoch

AUGMENTING THE LSTM PART-OF-SPEECH TAGGER WITH CHARACTER-LEVEL FEATURES

In the example above, each word had an embedding, which served as the inputs to our sequence model. Let’s augment the word embeddings with a representation derived from the characters of the word. We expect that this should help significantly, since character-level information like affixes have a large bearing on part-of-speech. For example, words with the affix -ly are almost always tagged as adverbs in English. See: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py

### Seq2seq translation (PT)

In this project we will be teaching a neural network to translate from French to English. An encoder network condenses an input sequence into a vector, and a decoder network unfolds that vector into a new sequence.

**Consider the sentence “Je ne suis pas le chat noir” → “I am not the black cat”. The words are in different order and different numbers of words, thus using ordinary one RNN is hard.**

#### Data Preprocessing

1. Read text file and split into lines, split lines into pairs
2. Normalize text, filter by length of words < 10 (no more than 10 words in one sentence; else hard to learn), delete pairs which contain words' occurance less than 3 times
3. We get list: `pairs`. eg: pairs[100]=['je pars .', 'i m going .']
4. Turn into tensor. eg: input sentence: [[ 6],[88],[ 5],[ 1]] (dim=[4,1]); output sentence: [[ 2],[ 3],[61],[ 4],[ 1]] (dim=[5,1]). 
The last word of every sentence is EOS, encoded as [1]. (In each language, we encode every word into a number. eg: ['je', 'pars', '.', 'EOS']).

#### Dimension Flow

batch=1. seq changes every sentence. If 'je pars .' then seq=4 (seperate by ' ' and add EOS). hidden_size = 256. input_lang.n_words=4489 (total 4489 different words in input language french). output_lang.n_words=2295.

##### Simplest seq2seq

1. Encoder: input [seq,1]. Each time use one of seq, [1] -> (`Embedding(4489,256)`) [1,256] -> (view) [1,1,256] -> (`GRU(256,256)`) 
output [1,1,256], state [1,1,256]. Recurrently use state. Save to output [seq,256] (`output[i]=hidden[0,0]`).

It is **wierd** why the orignial [Pytorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) use for loop each time of the seq. I think we can combine them together in below: 

1. Encoder: input [seq](a sentence) -> (`Embedding(4489,256)`) [seq,256] -> (view) [seq,1,256] -> (`GRU(256,256)`) 
output [seq,1,256], state [1,1,256].
2. Decoder: input [1,1] (exactly tensor([[0]]), which is SOS) -> (`Embedding(2295,256)`) [1,256] -> (view) [1,1,256] -> (`GRU(256,256)`) 
output [1,1,256], state [1,1,256]; output [1,1,256] -> (view) [1,256] -> (linear+softmax) [1,2295].
3. In encoder-decoder, the encoder_output is of no use but the encoder_state is used as initial decoder_state. It is called **context vector** as it encodes context from the entire sequence.
4. Now since initial decoder_state contains information of input_language, we could use decoder to get output sequentially.

##### Attention seq2seq

1. Encoder: the same as before, output [seq,1,256]. View and add 0 to turn output: [10,256] (max_length=10).
2. Decoder: input [1,1] (exactly tensor([[0]]), which is SOS) -> (`Embedding(2295,256)`) [1,256]. Denote as embedded.
3. Attension_weight: concate embedded [1,256] and state [1,256] into [1,512] -> (linear+softmax) [1,10].
4. Attension_applied: torch.bmm(attn_weights.unsqueeze(0) [1,1,10], encoder_outputs.unsqueeze(0) [1,10,256]) -> [1,1,256] (`bmm` is batch matrix-matrix product).
5. Output1: concate embedded [1,256] and attension_applied [1,256] into [1,512] -> (linear+relu) [1,256].
6. Output: output1 [1,256] -> (view+`gru(256,256)`) output [1,1,256] -> (view+softmax) [1,2295]. The initial hidden state of GRU is the last hidden state of the encoder.

#### Model structure

![](./pics/seq2seq1.jpg)

The above figures are model and encoder. The below figures are decoders.

![](./pics/seq2seq2.jpg)

#### Why it works?

Encoder step is the same.

simple: First embed `input` to `embeded` [1,256]. Use `embeded` and `hidden state` (initial at encoder's last hidden state) to do GRU. Then predict.

Flaw of simple Seq2seq: if only the context vector is passed betweeen the encoder and decoder, that single vector carries the burden of encoding the entire sentence. Loss information.

Attension: First embed `input` to `embeded`. Then we want to use both `embeded` [1,256] and `encoder_outputs` [10,256] as the input of GRU. After this, we do the same GRU procedure with `hidden state` as simple seq2seq.

However, encoder_outputs [10,256] is sequence of 10, thus we do multiplication to turn into `attension_applied` [1,256].

How to get the weight [1,10] of multiplication? We use `embeded` and `hidden state` to get `attention_weights` [1,10], which is the weights of `encoder_outputs` [10,256].

It gives me an inspiration: **First we determine input and the meaning of output. Then we just do Linear to realize it automatically. If there are several inputs, we concate them. If the dimension is not comply, we use Linear.** Quite strong and unreasonable.

#### LOSS

We know the target tensor (a sentence) and its length. Then we can run decoder target_length times, each time we can use NegativeLL to compute this current target word.

We can set to use Teacher forcing or not. “Teacher forcing” is the concept of using the real target outputs as each next input, instead of using the decoder’s guess as the next input. Using teacher forcing causes it to converge faster but when the trained network is exploited, it may exhibit instability.

```python
if not use_teacher_forcing:
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break
```

#### Evaluation

This time, we know nothing about target length. Thus, we do for loop for max_length (10) times. If current highest prob is EOS, we break.

After we guess the current Eng word as the highest prob, we also get attension (vector of length 10), which is the focus (weight) of the given Fren sentence (total max 10 words). It is interpretable and we may show it as a matrix after guessing all the Eng sentence.

#### What's more

1. Other datasets: Chat → Response, Question → Answer
2. Replace the embeddings with pre-trained word embeddings such as word2vec or GloVe



### Chatbot (PT)

We will train a simple chatbot using movie scripts from the Cornell Movie-Dialogs Corpus.

#### Data preprocessing

1. Read in movie scripts, seperate into sentences (Q & A), encode every words into numbers and trim words which occur less than 10 times.
2. Turn into tensor. Add EOS_token = 2.
3. This time, we use batch. However, since every sentence has different length, we padding 0 after EOS to make them the same length (set MAX_LENGTH = 10).
4. We add a `mask` tensor. The binary mask tensor has the same shape as the output target tensor, but every element that is a PAD_token is 0 and all others are 1. `mask=output!=0`.

Example1:

pairs[0]=['there .', 'where ?']

input:  batch2TrainData(voc, [pairs[0]])[0]=
tensor([[3],
        [4],
        [2]])
because EOS_token = 2

output: batch2TrainData(voc, [pairs[0]])[2]=
tensor([[5],
        [6],
        [2]])

mask: batch2TrainData(voc, [pairs[0]])[3]=
tensor([[1],
        [1],
        [1]])

Example2:

p=[pairs[0],pairs[1]]=[['there .', 'where ?'], ['you have my word . as a gentleman', 'you re sweet .']]

input: torch.Size([9, 2]); output: torch.Size([5, 2]); mask: torch.Size([5, 2])

output=tensor(

       [[ 7,  5],

        [14,  6],
        
        [15,  2],
        
        [ 4,  0],
        
        [ 2,  0]])
  
Thus, input & output: [seq,batch]. We fix batch_size=64. `seq` varies from every input & output. `seq` is the max length of words in this batch of setences and `seq` <=10.

#### Dimension flow

1. Input: input [seq,batch], hidden_state [2,batch,hidden] (bidirection GRU).
2. Encoder: input [seq,batch] -> (`embedding`) [seq,batch,hidden] -> (biGRU) [seq,batch,2\*hidden] -> (sum) encoder_output [seq,batch,hidden], hidden_state.
3. Decoder: input [1,batch] (exactly SOS) -> (`embedding`) [1,batch,hidden] -> (GRU) rnn_output [1,batch,hidden].
4. Attn_weights: `gloab_attn(rnn_output, encoder_outputs)` [seq,batch] (is the weights of each word).
5. Attension_applied: `torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))`  [batch,hidden].
6. Output1: concate rnn_output [batch,hidden] and attension_applied [batch,hidden] into [batch,2\*hidden] -> (linear+tanh) [batch,hidden] -> (linear+softmax) [batch,vocab].

There are two differences between chatbot tutorial and translation tutorial.

1. In translation, we do decoder's GRU at last, while in chatbot we do GRU just after embedded.
2. **Local attention**. Translation's attension_weight is calculated by embedded (rnn_output) and hidden_state: concate them and Linear to [seq].
3. **Global attention**. Chatbot's attension_weight is calculated by embedded (rnn_output) and encoder_outputs: dot/concate+Linear.

Model structure:

Here, blue box is the embedded input_sentence while red box is the embedded desired_output_word. We use both of the embedded to get global weight. Then it is the same as seq2seq translation: we bmm weight and embedded input_sentence to get attention (context). Concate context with embedded desired_output_word and Linear+softmax to probability.

Note: although there are two red box shown, each time we just use one box (one word).

![](./pics/chatbot_structure.jpg)


#### Loss

 Loss function calculates the average negative log likelihood of the elements that correspond to a 1 in the mask tensor.
 
 In order to converge, we do several tricks: using teacher forcing; gradient clipping.
 
 #### Save & load model
 
 Saving model:
 
 ```python
 encoder = EncoderRNN()
decoder = LuongAttnDecoderRNN()

# Save checkpoint
if (iteration % 1000 == 0):
    directory = os.path.join(save_dir, model_name, '{}-{}'.format(encoder_n_layers, decoder_n_layers))
    if not os.path.exists(directory): os.makedirs(directory)
    torch.save({
        'iteration': iteration,
        'en': encoder.state_dict(),
        'de': decoder.state_dict(),
        'en_opt': encoder_optimizer.state_dict(),
        'de_opt': decoder_optimizer.state_dict(),
        'loss': loss,
        'embedding': embedding.state_dict()
    }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))    
 ```

We can load model in another py file. We can train from then on or test our model.

```python
# Use checkpoint from 4000 iteration
checkpoint_iter = 4000
loadFilename = os.path.join(directory,'{}_checkpoint.tar'.format(checkpoint_iter))
checkpoint = torch.load(loadFilename)
# If loading a model trained on GPU to CPU
#checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
encoder_optimizer_sd = checkpoint['en_opt']


encoder = EncoderRNN()
encoder.load_state_dict(encoder_sd)
#encoder = encoder.to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
encoder_optimizer.load_state_dict(encoder_optimizer_sd)

# train or test as usual
```
