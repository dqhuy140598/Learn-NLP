import torch.nn as nn
import torch
from lang import SOS_token, EOS_token
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self, n_words, embedding_size, hidden_size, bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=n_words, embedding_dim=embedding_size)
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, bidirectional=bidirectional)

    def forward(self, x, hidden):
        seq_len = len(x)
        embedded = self.embedding(x)
        embedded = embedded.view(seq_len,1,-1)
        outputs, hidden = self.gru(embedded,hidden)
        return outputs, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(1, 1, self.hidden_size))
        return hidden

class Attn(nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method =="general":
            self.attn = nn.Linear(self.hidden_size,hidden_size)

        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size*2,hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(hidden_size))

    def score(self,hidden,encoder_output):
        if self.method == "dot":
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == "general":
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == "concat":
            energy = self.attn(torch.cat((hidden,encoder_output),dim=1))
            energy = self.other.dot(energy.squeeze(0))
            return energy

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = Variable(torch.zeros(seq_len))
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden,encoder_outputs[i])
        return torch.nn.functional.softmax(attn_energies).unsqueeze(0).unsqueeze(0)


class AttentionDecoder(nn.Module):

    def __init__(self, n_words, embedding_size, hidden_size, max_length, drop_out=0.1,attn_mode = "concat"):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_words = n_words
        self.embedding_size = embedding_size
        self.drop_out_p = drop_out
        self.max_length = max_length
        self.attn_mode = attn_mode

        self.embedding = nn.Embedding(num_embeddings=self.n_words,embedding_dim=embedding_size)
        self.gru = nn.GRU(input_size=hidden_size * 2, hidden_size= hidden_size, dropout= drop_out)
        self.out = nn.Linear(hidden_size * 2, n_words)

        self.attn = Attn(method=self.attn_mode,hidden_size = hidden_size)

    def forward(self, word_input , last_context , last_hidden, encoder_outputs):

        word_embedded = self.embedding(word_input)
        word_embedded = word_embedded.view(1,1,-1)
        decoder_input = torch.cat((word_embedded,last_context.unsqueeze(0)),dim=2)
        decoder_output, hidden = self.gru(decoder_input,last_hidden)
        attn_weights = self.attn(decoder_output.squeeze(0), encoder_outputs)
        attn_weights = attn_weights.to(torch.device('cuda:0'))
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        decoder_output = decoder_output.squeeze(0)
        context = context.squeeze(1)
        output = self.out(torch.cat((decoder_output,context),dim=1))
        output = torch.nn.functional.log_softmax(output,dim=1)
        return output, context, hidden , attn_weights

if __name__ == '__main__':
    max_length = 10
    n_words_1 = 100
    n_words_2 = 150
    hidden_size = 128
    embedding_size = hidden_size
    encoder = Encoder(n_words_1, embedding_size, hidden_size)
    decoder = AttentionDecoder(n_words_2,embedding_size,hidden_size,max_length)
    input_tensor = Variable(torch.LongTensor([1, 2, 3]))

    encoder_outputs,hidden = encoder(input_tensor)

    word_inputs = Variable(torch.LongTensor([1, 2, 3]))

    decoder_attns = torch.zeros([1,2,3])
    decoder_hidden = torch.zeros(size=(1,1,hidden_size))
    decoder_context = Variable(torch.zeros(1,hidden_size))

    for i in range(3):
        decoder_output,decoder_context,decoder_hidden, decoder_attn = decoder(word_inputs[i],decoder_context,\
                                                                                decoder_hidden,encoder_outputs)
        print(decoder_output.shape)

    print(encoder)
    print(decoder)

