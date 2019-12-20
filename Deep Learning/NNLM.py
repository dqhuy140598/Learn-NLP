import numpy as np
import torch
from torch.autograd import Variable

sentences = ['i like dog','i like coffee','i hate milk']

"""
N-Grams language model. it predicts the next word given a sequence of previous words
Paper: A Neural Probabilistic Language Model(2003)
"""


def build_vocab(sentences):
    word_vocab = []
    for sent in sentences:
        words = sent.split(" ")
        word_vocab.extend(words)
    word_dict = dict()
    word_dict["<PAD>"] = 0
    word_dict["<UNK>"] = 1
    count = 2
    for word in word_vocab:
        if word not in word_dict.keys():
            word_dict[word] = count
            count +=1

    number_dict = {v:k for k,v in word_dict.items()}
    return word_dict,number_dict


def build_dataset(sentences,word_dict):
    X_train = []
    y_train = []
    for sent in sentences:
        words = sent.split(" ")
        temp = []
        for word in words[:-1]:
            temp.append(word_dict[word])
        X_train.append(temp)
        y_train.append(word_dict[words[-1]])
    return X_train,y_train


class NeuralNetworkModel(torch.nn.Module):

    def __init__(self,n_step,embedding_size,n_hidden,n_classes):
        super(NeuralNetworkModel, self).__init__()
        self.n_step = n_step
        self.embedding_size = embedding_size
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.C = torch.nn.Embedding(num_embeddings=self.n_classes,embedding_dim=self.embedding_size)
        self.W = torch.nn.Parameter(torch.rand(size=(self.n_step * self.embedding_size,self.n_classes)))
        self.b = torch.nn.Parameter(torch.zeros(size=(self.n_classes,)))

        self.H = torch.nn.Parameter(torch.rand(size=(self.n_step*self.embedding_size,self.n_hidden)))
        self.d = torch.nn.Parameter(torch.rand(size=(self.n_hidden,)))

        self.U = torch.nn.Parameter(torch.rand(size=(self.n_hidden,self.n_classes)))

    def forward(self, input):
        X = self.C(input)
        X = X.view(-1,self.n_step*self.embedding_size).contiguous()
        output = self.b + torch.mm(X,self.W) + torch.mm(torch.tanh(torch.mm(X,self.H) + self.d),self.U)
        return output


word_dict, number_dict = build_vocab(sentences)


def train(epoch,lr,n_classes,n_step=2,embedding_size=50,n_hidden=64):

    X_train, y_train = build_dataset(sentences, word_dict)
    model = NeuralNetworkModel(n_step=n_step,n_classes=n_classes,
                               embedding_size=embedding_size,n_hidden=n_hidden)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),lr=lr)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    input_batch = torch.from_numpy(X_train).type(torch.LongTensor)
    input_label = torch.from_numpy(y_train).type(torch.LongTensor)

    for i in range(1,epoch):

        optimizer.zero_grad()
        output = model(input_batch)
        loss = loss_func(output,input_label)

        epoch_loss = loss.item()

        print("epoch:{0}, loss:{1:.2f}".format(i,epoch_loss))

        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1,keepdim=True)[1].numpy().tolist()
    for i,sent in enumerate(sentences):
        train = sent.split(" ")[:-1]
        word_pred = number_dict[predict[i][0]]
        print(train,'->',word_pred)


train(epoch=50,lr=0.01,n_classes=len(word_dict.keys()))

