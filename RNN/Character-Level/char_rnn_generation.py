# -*- coding: utf-8 -*-
"""
Generate names. [failed, I don't know why...]

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
"""
##############################################################################
#preprocessing
#word-dictionary: category_lines

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))

######################################################################
# Define input-raw


import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

category, line=randomTrainingPair()
#category: 'Italian';   line: 'Barone'

######################################################################
# Define input-tensor

import torch
import torch.nn as nn
# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)


def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

######################################################################
# test input
category_tensor=categoryTensor('Italian')
#   shape: [1, 18], one-hot
input_line_tensor=inputTensor('Barone')
#   shape: [6, 1, 59], in this example, there are 6 letters
target_line_tensor=targetTensor('Barone')
# targetTensor of 'Barone' should be 'aroneEOC'
#   tensor([ 0, 17, 14, 13,  4, 58]), here, 'a'=0, 'r'=17, 'e'=4, 'EOC'=58.
targetTensor('Aaron')
#   tensor([ 0, 17, 14, 13, 58])

######################################################################
# Creating the Network


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.layer_dim, nonlinearity='relu')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x,h0):
            
        # One time step
        out, hn = self.rnn(x,h0)
        out = self.fc(out[-1,:, :]) 
        #out=nn.Dropout(0.1)(out)
        return out,hn


n_hidden = 128
rnn = RNN(n_letters, n_hidden, 1,n_letters)
h0 = torch.zeros((1, 1, n_hidden))

input = inputTensor('Barone')
#torch.Size([6, 1, 59])
output,_ = rnn(input,h0)
print(output)
output.size()
#torch.Size([1, 59])

######################################################################

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


######################################################################
#train

import torch.optim as optim
learning_rate = 0.000005
optimizer=optim.SGD(rnn.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss()

rnn = RNN(n_letters, 128, 1,n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    
    category_tensor, input_line_tensor, target_line_tensor=randomTrainingExample()
    optimizer.zero_grad()

    loss = 0
    hn = torch.zeros((1, 1, n_hidden))

    for i in range(input_line_tensor.size(0)):
        output,hn= rnn(input_line_tensor[i].reshape((1,1,59)),hn)
        l = criterion(output, target_line_tensor[i].reshape(1))
        loss += l

    loss.backward()
    optimizer.step()

    loss=loss.item() / input_line_tensor.size(0)
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


import matplotlib.pyplot as plt
plt.figure()
plt.plot(all_losses)

######################################################################
# Sampling the Network

max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        input = inputTensor(start_letter)
        hidden = torch.zeros((1, 1, n_hidden))

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(input, hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')




