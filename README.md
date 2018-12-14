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

In CycleGAN and RNN, we don't define weights ourselves. We use `tf.layers.dense`, `tf.layers.conv2d` for CNN and 
`tf.nn.rnn_cell` or `tf.nn.static_rnn` for RNN.

If we want to update a subset of weights, we need to state layers' names.



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

## Recurrent Neural Network (RNN)
1. [Classifying names from languages] (PT)
2. [Generating names from languages] (PT)
3. [Predicting hand-written number] (TF)
4. [Generating shakespeares play] (TF)

RNN trains a hidden state (in LSTM trains several gates and cell state) and in each sequence (time step), we use both input and current hidden state to compute the next state. After that, we use a linear network to convey hidden state into output.

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
