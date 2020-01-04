import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

dtype = torch.FloatTensor

sentence = (
    'Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation'
)

word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
n_class = len(word_dict)
max_len = len(sentence.split())
n_hidden = 5


def make_batch(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_dict[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))
        target = word_dict[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))


class BiLSTM(nn.Module):

    def __init__(self):
        super(BiLSTM, self).__init__()

        self.bilstm = torch.nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)

        self.linear = torch.nn.Linear(in_features=2 * n_hidden, out_features=n_class)

    def forward(self, x):
        input = torch.transpose(x, dim0=0, dim1=1).contiguous()
        outputs, (_, _) = self.bilstm(input)
        outputs = outputs[-1]
        model = self.linear(outputs)
        return model


def train(epoch, lr):
    input_batch, target_batch = make_batch(sentence)
    model = BiLSTM()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for i in range(1, epoch):
        optimizer.zero_grad()
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        output = model(input_batch)
        loss = loss_func(output, target_batch)
        if (i + 1) % 100 == 0:
            print('Epoch:', '%d' % (i + 1), 'cost =', '{:.6f}'.format(loss.item()))
        loss.backward()
        optimizer.step()

    model.eval()
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(sentence)
    print([number_dict[n.item()] for n in predict.squeeze()])


train(1000, 0.001)
