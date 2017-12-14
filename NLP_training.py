import numpy as np
import sys
from matplotlib import pyplot

import embedding
import torch
from torch.autograd import Variable
import torch.nn as nn
import csv
from conll_df import conll_df

language = 'en'
file = 'lang_{}/gold/{}-ud-train.conllu'.format(language, language)

def prepare_data(file):
    # df = conll_df(file, file_index=False)
    # df_new = conll_df(file, file_index=False)[['w', 'x', 'g', 'f']]
    data_string = (conll_df(file, file_index=False)[['w', 'x', 'g', 'f']]).to_csv()

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

    # make a list of numpy 2D arrays from all the sentences
    sentences = []
    sent_idx = -1
    for i in range(len(properties['words'])):
        if (properties['idx'][i] == 1):
            if(properties['idx'][i+1] == 1): continue # skip one word sentences
            sent_idx +=1
            sentences.append(np.array(['<root>', '<root>', 0, 'root']))
            sentences[sent_idx] = np.vstack((sentences[sent_idx], np.array([properties['words'][i], properties['tags'][i], properties['dep_heads'][i], properties['labels'][i]])))
        else:
            sentences[sent_idx] = np.vstack((sentences[sent_idx], np.array([properties['words'][i], properties['tags'][i], properties['dep_heads'][i], properties['labels'][i]])))

    np.random.seed(0)
    np.random.shuffle(sentences)

    return sentences

def embed_sentence(sentence, language):

    batch_size = 1
    in_size = 125

    word_embeddings = embedding.get_word_embeddings(language)
    tag_embeddings = embedding.get_tag_embeddings(language)

    sentence_array = np.zeros((len(sentence), 125))
    for i in range(sentence.shape[0]):

        if not sentence[i, 0] in word_embeddings:
            sentence[i, 0] = '<unk>'

        sentence_array[i, :] = embedding.concatenate(word_embeddings[sentence[i, 0]], tag_embeddings[sentence[i, 1]])

    sentence_tensor = torch.from_numpy(sentence_array.astype(np.float32))
    sentence_tensor = sentence_tensor.view(len(sentence), batch_size, in_size)

    return sentence_tensor

def calc_gold(sentence):

    heads = []

    for line in sentence:
        heads.append(line[2])

    for i in range(len(sentence)):
        # word is root -> mark with arrow to itself
        if(heads[i] == "0"): heads[i] = i
        else: heads[i] = int(heads[i])-1

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
        concat = torch.cat((
                biLSTM_embed.repeat(sentence_emb.size()[0],1,1),
                biLSTM_embed.repeat(1,1,sentence_emb.size()[0]).view(-1,1,250)
        ), 2)
        inp = concat.view(-1, 500)
        out = self.MLP(inp)
        output_mtx = out.view(sentence_emb.size()[0], -1)
        return output_mtx

# Hyperparameter
lstm_in_size = 125
lstm_h_size = 125
lstm_num_layers = 1

MLP_D_in = 500
MLP_D_H = int(sys.argv[1])
MLP_D_out = 1

learning_rate = float(sys.argv[2])
loss_criterion = nn.CrossEntropyLoss()
model = LSTMParser()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TRAINING
def train_step(model, input_sent, goldtree, loss_criterion, optimizer):

    model.zero_grad()
    output_mtx = model(input_sent)
    loss = loss_criterion(output_mtx, goldtree.view(goldtree.size()[1]))
    loss.backward()
    optimizer.step()

    return output_mtx, loss


# define the training for loop
def train(filename, model, language, verbose = 1):

    sentences = prepare_data(filename)

    losses = []

    for epoch in range(int(sys.argv[3])):

        epoch_loss = 0

        if verbose > 0: print('\n***** Epoch {}:'.format(epoch))

        for sentence in sentences:

            sentence_var = Variable(embed_sentence(sentence, language), requires_grad=False)

            gold = Variable(calc_gold(sentence))
            parse_mtx, loss = train_step(model, sentence_var, gold, loss_criterion, optimizer)
            epoch_loss += loss

            if verbose > 1: print('loss {0:.4f} for "'.format(loss.data.numpy()[0]) + ' '.join(word for word in sentence[:,0]) + '"')

        torch.save(model.state_dict(), "lang_{}/models/{}.pth".format(language, sys.argv[4]))
        losses.append(epoch_loss.data.numpy()[0] / len(sentences))

        if verbose > 0: print('average loss {} \n*****'.format(losses[-1]))

        pyplot.plot(range(len(losses)), losses)
        pyplot.savefig('lang_{}/models/{}_loss.pdf'.format(language, sys.argv[4]))


if __name__ == "__main__":
    train(file, model, language)
