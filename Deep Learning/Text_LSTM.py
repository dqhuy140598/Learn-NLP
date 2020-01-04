import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
word_dict = {n: i for i, n in enumerate(char_arr)}
number_dict = {i: w for i, w in enumerate(char_arr)}
n_class = len(word_dict) # number of class(=number of vocab)

seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

n_step = 3
n_hidden = 128


def make_batch(seq_data):
    input_batch, target_batch = [], []
    for seq in seq_data:
        input = [word_dict[n] for n in seq[:-1]] # 'm', 'a' , 'k' is input
        target = word_dict[seq[-1]] # 'e' is target
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))


class TextLSTM(nn.Module):

    def __init__(self):
        super(TextLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class,hidden_size=n_hidden)
        self.linear = nn.Linear(in_features=n_hidden,out_features=n_class)

    def forward(self, x):
        input = torch.transpose(x,dim0=0,dim1=1).contiguous()
        output,(_,_) = self.lstm(input)
        output = output[-1]
        model = self.linear(output)
        return model


def train(epoch,lr):

    input_batch, target_batch = make_batch(seq_data)

    model = TextLSTM()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    for i in range(1,epoch+1):

        optimizer.zero_grad()
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        output = model(input_batch)
        loss = loss_func(output,target_batch)

        if (i + 1) % 100 == 0:
            print('Epoch:', '%04d' % (i + 1), 'cost =', '{:.6f}'.format(loss.item()))

        loss.backward()
        optimizer.step()

    model.eval()

    inputs = [sen[:3] for sen in seq_data]

    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])


train(1000,0.001)