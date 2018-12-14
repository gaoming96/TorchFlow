# -*- coding: utf-8 -*-
"""
The dataset is a period of shakespeares novel. We set batch=1, seq=25 (25 letters). 
We generate new chars after given the beginning of words.

#### Flow:
1. Input: (1,25,65). Totally 65 different chars in this novel. Output: (1,25,65), 
which is input moving forward a letter. eg: `Gaomin` is the input, then `aoming` is the output.

2. `tf.unstack` to 25@[1,65], then we RNN -> hidden: 25@[1,100].

3. In each hidden[i], we define linear network (reuse weight for all time step), 
to get logit: 25@[1,65].

4. We compute loss of logit: 25@[1,65] and output: 25@[1,65] (sum of all time step).

5. In train step, we use the current hidden state as the input hidden state in each epic 
(trained all chars one time), and then clear to 0 in next epic.

6. For test step, we first give 25 letters. Then each step, we choose the highest 
prob of the 26th letter. After that, we abandon the first letter and choose 2-26th 
letters as the input and repeat. We use current hidden state and reuse hidden-output as hidden-input.



"""
###############################################################################
# input

import tensorflow as tf
import numpy as np

text = open('input.txt', 'r').read() # should be simple plain text file
uniqueChars = list(set(text))
text_size, vocab_size = len(text), len(uniqueChars)
print('data has %d characters, %d unique.' % (text_size, vocab_size))
# data has 1115393 characters, 65 unique.
char_to_ix = { ch:i for i,ch in enumerate(uniqueChars) }
ix_to_char = { i:ch for i,ch in enumerate(uniqueChars) }

hidden_size = 100
seq_length = 25
def one_hot(v):
    return np.eye(vocab_size)[v]

###############################################################################
# test input

positionInText      = 2018
inputs = one_hot([char_to_ix[ch] for ch in text[positionInText:positionInText+seq_length]])
targets = one_hot([char_to_ix[ch] for ch in text[positionInText+1:positionInText+seq_length+1]])
# (25, 65). seq=25, 25 chars each time.
inputs=inputs.reshape((1,seq_length,65))
targets=targets.reshape((1,seq_length,65))

###############################################################################
# model

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [1,None,vocab_size])
y_in = tf.placeholder(tf.float32, [1,None,vocab_size])
hStart = tf.placeholder(tf.float32,[1,hidden_size])

X_list= tf.unstack(x, seq_length, 1)
labels_series=tf.unstack(y_in, seq_length, 1)

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
outputs, states = tf.nn.static_rnn(rnn_cell, X_list, initial_state=hStart,dtype=tf.float32)


initializer = tf.random_normal_initializer(stddev=0.1)

with tf.variable_scope("RNN",reuse=tf.AUTO_REUSE) as scope:
    logits_series = []
    for i in range(seq_length):
        
        tmp = tf.layers.dense(outputs[i], units=vocab_size, activation=None,
                         kernel_initializer=initializer)
        
        logits_series.append(tmp)

# used in test
hLast = states
output_softmax = tf.nn.softmax(logits_series[-1])

losses = [tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) 
            for logits, labels in zip(logits_series,labels_series)]
loss = tf.reduce_mean(losses)


#we can't simply optimize using the gradients noramlly computer. 
#Left by themselves, the gradients in an RNN will increase exponentially, 
#and the network will fail to converge. 
#In order to deal with this, we clip them within the range of -5 to 5.
minimizer = tf.train.AdamOptimizer(learning_rate=0.003)
grads_and_vars = minimizer.compute_gradients(loss)

grad_clipping = tf.constant(5.0, name="grad_clipping")
clipped_grads_and_vars = []
for grad, var in grads_and_vars:
    clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
    clipped_grads_and_vars.append((clipped_grad, var))

updates = minimizer.apply_gradients(clipped_grads_and_vars)

###############################################################################
# test
import random
def sampleNetwork():
    sample_length = 200
    start_ix      = random.randint(0, len(text) - seq_length)
    sample_seq_ix = [char_to_ix[ch] for ch in text[start_ix:start_ix + seq_length]]
    ixes          = []
    sample_prev_state_val = np.copy(hStart_val)

    for t in range(sample_length):
        sample_input_vals = one_hot(sample_seq_ix)
        sample_input_vals=sample_input_vals.reshape((1,25,65))
        sample_output_softmax_val, sample_prev_state_val = \
        sess.run([output_softmax, hLast], feed_dict={x: sample_input_vals, hStart: sample_prev_state_val})

        ix = np.random.choice(range(vocab_size), p=sample_output_softmax_val.ravel())
        ixes.append(ix)
        sample_seq_ix = sample_seq_ix[1:] + [ix]

    txt = ''.join(ix_to_char[ix] for ix in ixes)
    print('----\n %s \n----\n' % (txt,))
    
###############################################################################
# train
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

positionInText = 0
numberOfIterations = 0
totalIterations = 10000
hStart_val = np.zeros([1,hidden_size])

while numberOfIterations < totalIterations:
    if positionInText+seq_length+1 >= len(text) or numberOfIterations == 0: 
        hStart_val = np.zeros([1,hidden_size])
        positionInText = 0
        
    inputs = one_hot([char_to_ix[ch] for ch in text[positionInText:positionInText+seq_length]])
    targets = one_hot([char_to_ix[ch] for ch in text[positionInText+1:positionInText+seq_length+1]])
    inputs=inputs.reshape((1,25,65))
    targets=targets.reshape((1,25,65))
    
    hStart_val, loss_val, _ = sess.run([hLast,loss,updates],feed_dict={x:inputs, y_in:targets,hStart:hStart_val})
    
    if numberOfIterations % 500 == 0:
        print('iter: %d, p: %d, loss: %f' % (numberOfIterations, positionInText, loss_val))
        sampleNetwork()
    
    positionInText += seq_length
    numberOfIterations +=1