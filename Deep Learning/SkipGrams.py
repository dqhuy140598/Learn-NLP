import numpy as np
import torch
import matplotlib.pyplot as plt
sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

window_size = 1


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


word_dict,number_dict = build_vocab(sentences)
n_classes = len(word_dict.keys())


def build_dataset(sentences):
    skip_grams = []
    for sent in sentences:
        words = sent.split(" ")
        for i in range(1,len(words) - 1):
            target = word_dict[words[i]]
            context = [word_dict[words[i-window_size]], word_dict[words[i + window_size]]]
            for w in context:
                skip_grams.append([target,w])

    return skip_grams


skip_grams = build_dataset(sentences)


def shuffle_data(data):
    shuffle_index = np.random.permutation(data.shape[0])
    data_shuffle = data[shuffle_index]
    return data_shuffle


def get_batch(data,batch_size):
    random_index = np.random.choice(data.shape[0],size=batch_size,replace=False)
    input_batch = []
    input_label = []
    for i in random_index:
        input_batch.append(np.eye(n_classes)[data[i][0]])
        input_label.append(data[i][1])
    return np.array(input_batch), np.array(input_label)


class SkipGramModel(torch.nn.Module):

    def __init__(self,embedding_dim,n_classes):
        super(SkipGramModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.W = torch.nn.Parameter(2 * - torch.rand(size=(self.n_classes,self.embedding_dim)) + 1)
        self.WT = torch.nn.Parameter(2 * - torch.rand(size=(self.embedding_dim,self.n_classes)) + 1)

    def forward(self, X):
        embedding_vec = torch.mm(X,self.W)
        output = torch.mm(embedding_vec,self.WT)
        return output


def train(n_classes,epoch,lr,batch_size=4,embedding_dim=2):

    shuffled_data = shuffle_data(np.array(skip_grams))

    model = SkipGramModel(embedding_dim=embedding_dim,n_classes=n_classes)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)

    for i in range(1,epoch+1):

        optimizer.zero_grad()

        input_batch,input_labels = get_batch(shuffled_data,batch_size=batch_size)
        input_batch = torch.from_numpy(input_batch).type(torch.FloatTensor)
        input_labels = torch.from_numpy(input_labels).type(torch.LongTensor)

        output = model(input_batch)

        loss = loss_func(output,input_labels)

        epoch_loss = loss.item()
        if (i + 1) % 1000 == 0:
            print("epoch:{0}, loss:{1:.2f}".format(i+1, epoch_loss))
        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_dict):
        W, WT = model.parameters()
        x, y = float(W[i][0]), float(W[i][1])
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()


train(n_classes=n_classes,epoch=5000,lr=0.001)


