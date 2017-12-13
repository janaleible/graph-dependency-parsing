import numpy as np
from matplotlib import pyplot

import embedding
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
import csv
from conll_df import conll_df

language = 'en'
file = 'lang_{}/gold/en-ud-train.conllu'.format(language)

def prepare_data(file):
    df = conll_df(file, file_index=False)
    df_new = df[['w', 'x', 'g', 'f']]
    data_string = df_new.to_csv()

    data_list = list(csv.reader(data_string.split('\n')))[:-1]

    data_list.pop(0)
    properties = {"idx" : [], "words": [], "tags": [], "dep_heads":[], "labels":[]}
    for j in range(len(data_list)):
        tokens = data_list[j]
        properties['idx'].append(int(float(tokens[1])))
        properties['words'].append(tokens[2].lower())
        properties['tags'].append(tokens[3].lower())
        properties['dep_heads'].append(tokens[4])
        properties['labels'].append(tokens[5].lower())

    idx = properties['idx']
    words = properties['words']
    tags = properties['tags']
    dep_heads = properties['dep_heads']
    labels = properties['labels']

    # idx.pop(0)
    # words.pop(0)
    # tags.pop(0)
    # dep_heads.pop(0)
    # labels.pop(0)


    # make a list of numpy 2D arrays from all the sentences
    sentences = []
    sent_idx = -1
    for i in range(len(words)):
        if (idx[i] == 1):
            if(idx[i+1] == 1): continue
            # sentences.append(np.array("Root"))
            sentences.append(np.array([words[i], tags[i], dep_heads[i], labels[i]]))
            sent_idx +=1
        else:
            sentences[sent_idx] = np.vstack((sentences[sent_idx], np.array([words[i], tags[i], dep_heads[i], labels[i]])))




    np.random.seed(0)
    np.random.shuffle(sentences)

    return sentences

def embed_sentence(s):
    # Make the embedding for the selected sentence

    batch_size = 1
    in_size = 125

    sentence_array = np.zeros((len(s), 125))
    for i in range(s.shape[0]):
        try:
            sentence_array[i, :] = embedding.concatenate(embedding.embed_word[s[i, 0]], embedding.embed_tag[s[i, 1]])

        except(KeyError):
            sentence_array[i, :] = embedding.concatenate(embedding.embed_word["<unk>"],embedding.embed_tag[s[i, 1]])


    sentence_array = sentence_array.astype(np.float32)
    sentence_tensor = torch.from_numpy(sentence_array)
    # sent_float_tensor = torch.FloatTensor(sentence_tensor)
    sentence_tensor = sentence_tensor.view(len(s), batch_size, in_size)

    return sentence_tensor

def calc_gold(s):
    words = []
    tags = []
    heads = []
    relations = []
    for line in s:
        words.append(line[0])
        tags.append(line[1])
        heads.append(line[2])
        relations.append(line[3])
    dim = len(s)
    for i in range(dim):
        if(heads[i] == "0"):
            # word is root -> mark with arrow to itself
            heads[i]= int(i)
        else:
            heads[i] = int(heads[i])-1
    target = torch.from_numpy(np.array([heads]))
    return target




# Defining the LSTMParser class

class LSTMParser(nn.Module):
    """
    This class implements the whole parsing procedure
    """

    def __init__(self):
        super(LSTMParser, self).__init__()
        # self.weights = Parameter()

        self.biLSTM = nn.LSTM(
            input_size=lstm_in_size,
            hidden_size=lstm_h_size,
            num_layers=lstm_num_layers,
            bidirectional=True
        )

        self.MLP = nn.Sequential(
            torch.nn.Linear(MLP_D_in, MLP_D_H),
            torch.nn.Tanh(),
            torch.nn.Linear(MLP_D_H, MLP_D_out)
        )



    def forward(self, sentence_emb):
        biLSTM_embed, _ = self.biLSTM(sentence_emb)
        concat = torch.cat((biLSTM_embed.repeat(1,1,sentence_emb.size()[0]).view(-1,1,250),biLSTM_embed.repeat(sentence_emb.size()[0],1,1)), 2)
        inp = concat.view(-1, 500)
        out = self.MLP(inp)
        output_mtx = out.view(sentence_emb.size()[0], -1)
        return output_mtx

# Hyperparameter
lstm_in_size = 125
lstm_h_size = 125
lstm_num_layers = 1

MLP_D_in = 500
MLP_D_H = 100
MLP_D_out = 1

learning_rate = 1e-4
loss_criterion = nn.CrossEntropyLoss()
model = LSTMParser()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TRAINING
def train_step(model, input_sent, goldtree, loss_criterion, optimizer):
    model.zero_grad()
    output_mtx = torch.transpose(model(input_sent), 0, 1)
    loss = loss_criterion(output_mtx, goldtree.view(goldtree.size()[1]))
    loss.backward()
    optimizer.step()

    return output_mtx, loss


# define the training for loop
def train(filename, model, language, verbose = 1):

    sentences = prepare_data(filename)

    losses = []

    for epoch in range(10):

        epoch_loss = 0

        if verbose > 0: print('\n***** Epoch {}:'.format(epoch))

        for sentence in sentences:

            sentence_var = Variable(embed_sentence(sentence), requires_grad=False)

            gold = Variable(calc_gold(sentence))
            parse_mtx, loss = train_step(model, sentence_var, gold, loss_criterion, optimizer)
            epoch_loss += loss

            if verbose > 1: print('loss {0:.4f} for "'.format(loss.data.numpy()[0]) + ' '.join(word for word in sentence[:,0]) + '"')

        torch.save(model.state_dict(), "lang_{}/models/my_model.pth".format(language))
        losses.append(epoch_loss.data.numpy()[0] / len(sentences))

        if verbose > 0: print('average loss {} \n *****'.format(losses[-1]))

        pyplot.plot(range(len(losses)), losses)
        pyplot.savefig('lang_{}/models/my_model_loss.pdf'.format(language))


if __name__ == "__main__":
    train(file, model, language)
