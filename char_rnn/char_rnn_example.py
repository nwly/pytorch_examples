import unidecode
import string
import random
import re
import time, math

import torch
import torch.nn as nn
from torch.autograd import Variable

all_characters = string.printable
n_characters = len(all_characters)

file_str = unidecode.unidecode(open('input.txt').read())
file_len = len(file_str)


chunk_len = 50

### Data Preprocessing

def random_chunk(string_in=file_str, file_len=file_len, chunk_len=chunk_len):
    start_idx = random.randint(0, file_len - chunk_len)
    end_idx = start_idx + chunk_len + 1
    return string_in[start_idx:end_idx]

# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

def random_training_set():
    "Returns a tuple (x, y), with y leading x by 1 slot"
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target


### Model

class rnnModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, batch_size=1):
        # input_size = vocab_size
        # output_size = vocab_size
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        return (Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),
        Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)))
        # return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))  # 1 = batch_size

    def forward(self, x, hidden):
        x = self.encoder(x.view(1, -1))
        output, (hidden, lstm_cell) = self.lstm(x.view(1, 1, -1), hidden)
        # output, hidden = self.gru(x.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        # return output, hidden              # GRU
        return output, (hidden, lstm_cell)   # LSTM

### Training Parameters

n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

model = rnnModule(n_characters, hidden_size, n_characters, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


### Train + Eval functions

def evaluate(model_, prime_str='A', predict_len=100, temperature=0.8):
    """ev"""
    hidden = model_.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model_(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = model_(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)
    return predicted


def train(model_, model_optimizer, inp, target, chunk_len=chunk_len):
    hidden = model_.init_hidden()
    model_.zero_grad()
    loss = 0

    for c in range(chunk_len-1):
        output, hidden = model_(inp[c], hidden)
        loss += criterion(output, target[c].view(1))

    loss.backward()
    model_optimizer.step()

    return loss.data[0] / chunk_len

### Train model loop

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

all_losses = []
loss_avg = 0
start = time.time()

for epoch in range(1, n_epochs + 1):
    loss = train(model, optimizer, *random_training_set())
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate(model, 'Wh', 100), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

print(evaluate(model, 'Th', chunk_len, temperature=0.8))
