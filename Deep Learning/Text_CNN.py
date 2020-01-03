import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as f
dtype = torch.FloatTensor
embedding_size = 2
sequence_length = 3
num_classes = 2
filter_sizes = [2,2,2]

num_filters = 3

sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))

word_dict = {w:i for i,w in enumerate(word_list)}

vocab_size = len(word_dict)

inputs = []
for sen in sentences:
    inputs.append(np.asarray([word_dict[n] for n in sen.split()]))

targets = []
for out in labels:
    targets.append(out) # To using Torch Softmax Loss function

input_batch = Variable(torch.LongTensor(inputs))
target_batch = Variable(torch.LongTensor(targets))


class TextCNN(nn.Module):

        def __init__(self,device):
            super(TextCNN, self).__init__()

            self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_size).to(device)

            self.convs = [torch.nn.Conv2d(in_channels=1,out_channels=num_filters,kernel_size=(embedding_size,x)).to(device) for x in filter_sizes]

            self.flat_size = num_filters * len(filter_sizes)

            self.out = torch.nn.Linear(self.flat_size,num_classes).to(device)

        def forward(self, x):

            embeddings_input = self.embeddings(x)

            embeddings_input = torch.unsqueeze(embeddings_input,dim=1)

            output_pooled = []

            for conv in self.convs:

                output_conv = conv(embeddings_input)
                output_conv = torch.relu(output_conv)
                pool_window = list(output_conv.shape)[2]
                pooled = torch.nn.functional.max_pool2d(input=output_conv,kernel_size=(pool_window,1))
                pooled = pooled.permute(0,3,2,1).contiguous()

                output_pooled.append(pooled)

            output_pooled = torch.cat(output_pooled,dim=-1)
            output_pooled_flat = output_pooled.view(size=(-1,self.flat_size))

            output = self.out(output_pooled_flat)

            return output


def train(epoch,lr):

    global input_batch,target_batch

    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    else:
        device = torch.device('cpu')
    print("device: ",device)
    model = TextCNN(device)
    optimizer = optim.Adam(params=model.parameters(),lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for i in range(1,1000):
        optimizer.zero_grad()
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        output = model(input_batch)
        loss = loss_func(output,target_batch)
        if (i + 1) % 100 == 0:
            print('Epoch:', '%d' % (i + 1), 'cost =', '{:.6f}'.format(loss.item()))
        loss.backward()
        optimizer.step()

    model.eval()

    test_text = 'i love you'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = Variable(torch.LongTensor(tests)).to(device)

    # Predict
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")


train(5,0.001)