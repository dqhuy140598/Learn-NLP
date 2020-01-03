import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)

# TextRNN Parameter
batch_size = len(sentences)
n_step = 2 # number of cells(= number of Step)
n_hidden = 5 # number of hidden units in one cell


#one hot encoding
def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch


# to Torch.Tensor

input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.Tensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))


class TextRNN(torch.nn.Module):

        def __init__(self):
            super(TextRNN, self).__init__()
            self.rnn = torch.nn.RNN(input_size=n_class,hidden_size=n_hidden)

            self.linear = torch.nn.Linear(in_features=n_hidden,out_features=n_class)

        def forward(self, x):

            x = torch.transpose(x,dim0=0,dim1=1)

            outputs,hiddens = self.rnn(x)

            outputs = outputs[-1]

            model = self.linear(outputs)

            return model


def train(epoch,lr):

    global input_batch,target_batch

    model = TextRNN()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.train()

    for i in range(1,epoch+1):

        optimizer.zero_grad()
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        output = model(input_batch)

        loss = loss_func(output,target_batch)

        if (i+ 1) % 100 == 0:
            print('Epoch:', '%04d' % (i + 1), 'cost =', '{:.6f}'.format(loss.item()))

        loss.backward()
        optimizer.step()

    model.eval()
    input = [sen.split()[:2] for sen in sentences]

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])


train(1000,lr=0.001)