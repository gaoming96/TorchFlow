""" 
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.

### Predicting hand-written number

We set batch_size=64. Since a figure is 28\*28, we set seq=28 (each row of a figure) 
and input_dim=28. In this example, seq_length is fixed while in the above, seq depends on words.

#### Flow:
2. Input: In each round, 64 figures and their numbers.
3. [batch=64,seq=28, input_dim=28]-> (`tf.unstack`) 28@[batch=64,input_dim=28] 
-> hidden: 28@[batch=64, hid_dim=128].
Then we use hidden[-1] to do linear network -> output:[1,10].
Finally, we use output and number (one hot) to compute loss.

**Note that in each time step, we feed in input[i] to get hidden[i].**

**Note that hidden[-1] is the last time step, which can be regard as information 
we learnt from all the sequence (28 time steps).**

"""

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 64
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

###############################################################################
# tf Graph input

tf.reset_default_graph()
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

###############################################################################
# Model

# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
X_list = tf.unstack(X, timesteps, 1)

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)
outputs, states = tf.nn.static_rnn(lstm_cell, X_list, dtype=tf.float32)

logits = tf.layers.dense(outputs[-1], units=num_classes, activation=None,
                         kernel_initializer=tf.random_normal_initializer(seed=0))

prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

###############################################################################
# train

init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 16 mnist test images
    test_len = 16
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
