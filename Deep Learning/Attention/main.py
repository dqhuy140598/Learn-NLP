import random
from utils import filter_pairs, MAX_LENGTH
from lang import read_langs,SOS_token,EOS_token
from model import Encoder, AttentionDecoder
import torch.optim as optim
import torch.nn as nn
import torch
import time
import math
from torch.autograd import Variable
teacher_forcing_ratio = 0.5
clip = 5.0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence, use_cuda=True):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
    if use_cuda: var = var.to(device)
    return var


def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)


def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


def train(input_variable,target_variable,encoder,decoder,encoder_optimizer,
          decoder_optimizer,criterion,max_length=MAX_LENGTH,use_cuda=True):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    input_length = input_variable.size(0)
    target_length = target_variable.size(0)

    encoder_hidden = encoder.init_hidden()

    if use_cuda:
        encoder_hidden = encoder_hidden.to(device)

    encoder_outputs, encoder_hidden = encoder(input_variable,encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1,decoder.hidden_size))

    decoder_hidden = encoder_hidden

    if use_cuda:
        decoder_input = decoder_input.to(device)
        decoder_context = decoder_context.to(device)
        encoder_outputs = encoder_outputs.to(device)

    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:

        for di in range(target_length):
            decoder_output,decoder_context,decoder_hidden,decoder_attention = decoder(decoder_input,
                                                                                      decoder_context,
                                                                                      decoder_hidden,
                                                                                      encoder_outputs)
            loss += criterion(decoder_output,target_variable[di])

            decoder_input = target_variable[di]
    else:

        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                         decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss+= criterion(decoder_output,target_variable[di])

            topv,topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))

            if use_cuda:
                decoder_input = decoder_input.to(device)

            if ni == EOS_token: break

    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(),clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(),clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

    attn_model = 'concat'
    hidden_size = 128
    dropout_p = 0.05

    # Initialize models
    encoder = Encoder(n_words=input_lang.n_words, embedding_size= hidden_size, hidden_size =hidden_size)
    decoder = AttentionDecoder(n_words=output_lang.n_words,embedding_size=hidden_size,hidden_size=hidden_size,max_length=MAX_LENGTH,drop_out=dropout_p,attn_mode=attn_model)

    # Move models to GPU
    if use_cuda:
        encoder.to(device)
        decoder.to(device)

    # Initialize optimizers and criterion
    learning_rate = 0.0001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    n_epochs = 50000
    plot_every = 200
    print_every = 1000

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for epoch in range(1, n_epochs + 1):

        # Get training data for this cycle
        training_pair = variables_from_pair(random.choice(pairs))
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        input_variable = input_variable.to(device)
        target_variable = target_variable.to(device)

        # Run the train function
        loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
            time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    torch.save(encoder.state_dict(),'model/encoder.pth')
    torch.save(decoder.state_dict(),'model/decoder.pth')