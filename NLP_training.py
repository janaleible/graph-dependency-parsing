import numpy as np
import sys
from matplotlib import pyplot

import embedding
import torch
from torch.autograd import Variable
import torch.nn as nn
import csv
import pickle
from conll_df import conll_df

from mst import mst


def prepare_data(file, training=True):

    data_string = (conll_df(file, file_index=False)[['w', 'x', 'g', 'f']]).to_csv()

    data_list = list(csv.reader(data_string.split('\n')))[:-1]

    data_list.pop(0)
    properties = {"idx" : [], "words": [], "tags": [], "dep_heads":[], "labels":[]}
    for j in range(len(data_list)):
        tokens = data_list[j]
        properties['idx'].append(int(float(tokens[1])))
        properties['words'].append(tokens[2])
        properties['tags'].append(tokens[3].lower())
        properties['dep_heads'].append(tokens[4])
        properties['labels'].append(tokens[5].lower())

    # make a list of numpy 2D arrays from all the sentences
    sentences = []
    sent_idx = -1
    for i in range(len(properties['words'])):
        if (properties['idx'][i] == 1):
            sent_idx +=1
            sentences.append(np.array(['<root>', '<root>', 0, 'root']))
            sentences[sent_idx] = np.vstack((sentences[sent_idx], np.array([properties['words'][i], properties['tags'][i], properties['dep_heads'][i], properties['labels'][i]])))
        else:
            sentences[sent_idx] = np.vstack((sentences[sent_idx], np.array([properties['words'][i], properties['tags'][i], properties['dep_heads'][i], properties['labels'][i]])))

    np.random.seed(0)
    if training: np.random.shuffle(sentences)

    return sentences

def embed_sentence(sentence, language):

    batch_size = 1
    in_size = 125

    word_embeddings = embedding.get_word_embeddings(language)
    tag_embeddings = embedding.get_tag_embeddings(language)

    sentence_array = np.zeros((len(sentence), 125))
    for i in range(sentence.shape[0]):

        if not sentence[i, 0] in word_embeddings:
            embeddable = '<unk>'
        else: embeddable = sentence[i,0].lower()

        sentence_array[i, :] = embedding.concatenate(word_embeddings[embeddable], tag_embeddings[sentence[i, 1]])

    sentence_tensor = torch.from_numpy(sentence_array.astype(np.float32))
    sentence_tensor = sentence_tensor.view(len(sentence), batch_size, in_size)

    return sentence_tensor

def calc_gold_arcs(sentence):

    heads = []

    for line in sentence:
        heads.append(int(line[2]))

    target = torch.from_numpy(np.array([heads]))

    return target


class LSTMParser(nn.Module):
    """
    This class implements the whole parsing procedure
    """

    def __init__(self):
        super(LSTMParser, self).__init__()

        self.biLSTM = nn.LSTM(
            input_size=lstm_in_size,
            hidden_size=lstm_h_size,
            num_layers=lstm_num_layers,
            bidirectional=True
        )

        # for predicting arcs
        self.arcMLP = nn.Sequential(
            torch.nn.Linear(MLP_in, MLP_score_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(MLP_score_hidden, MLP_score_out)
        )

        # for predicting labels
        self.labelMLP = torch.nn.Sequential(
            torch.nn.Linear(MLP_in, MLP_label_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(MLP_label_hidden, MLP_label_out)
        )

    def forward(self, sentence_emb, gold_tree=None):

        biLSTM_embed, _ = self.biLSTM(sentence_emb)

        # stuff for scores
        arcs_in = torch.cat((
                biLSTM_embed.repeat(sentence_emb.size()[0],1,1),
                biLSTM_embed.repeat(1,1,sentence_emb.size()[0]).view(-1,1,250)
        ), 2).view(-1, 500)

        arc_scores = self.arcMLP(arcs_in).view(sentence_emb.size()[0], -1)

        if gold_tree is None:
            gold_tree = self.get_tree(arc_scores)

        # stuff for labels
        heads = biLSTM_embed
        dependants = Variable(torch.index_select(
            biLSTM_embed.data,
            0,
            torch.from_numpy(np.argmax(gold_tree, 1))
        ))

        label_matrix = self.labelMLP(torch.cat((heads, dependants), 2).view(-1, 500))

        return arc_scores, label_matrix

    def get_tree(self, scores_matrix):

        softmax = nn.Softmax()
        prediction = softmax(scores_matrix)

        prediction = prediction.data.numpy()

        # we represent the final parse as a words*words mtx, where the root is indicated as the diagonal element
        return mst(prediction)

    def predict(self, sentence):

        arc_matrix, label_matrix = self.forward(sentence)
        max_tree = self.get_tree(arc_matrix)

        return max_tree, label_matrix


def calc_gold_labels(sentence):

    labels = []

    with open('lang_en/embeddings/label2i.pickle', 'rb') as file:
        label2i = pickle.load(file)

    for line in sentence:
        labels.append(label2i[line[3]])

    target = torch.from_numpy(np.array([labels]))

    return target

def train_step(model, input_sent, gold_arcs, gold_labels, arc_loss_criterion, label_loss_criterion, optimizer):

    model.zero_grad()
    arc_matrix, label_matrix = model(input_sent)
    arc_loss = arc_loss_criterion(arc_matrix, gold_arcs.view(gold_arcs.size()[1]))
    label_loss = label_loss_criterion(label_matrix, gold_labels.view(gold_labels.size()[1]))
    loss = arc_loss + label_loss
    loss.backward()
    optimizer.step()

    del label_matrix

    return loss, arc_matrix, (arc_loss, label_loss)


def visualise_sentence(sentence, matrix, epoch, modelname, language):

    if len(sentence) > 6 and (
        all(sentence[:, 0][1:5] == ['The', 'third', 'was', 'being'])
        or all(sentence[:, 0][1:5] == ['Die', 'Soldaten', 'hätten', 'sowieso'])
    ):

        with open('lang_{}/models/sample_sentence/model_{}_epoch{}.pickle'.format(language, modelname, epoch), 'wb') as file:
            pickle.dump(matrix, file)


def train(filename, model, language, epochs, verbose = 2):

    sentences = prepare_data(filename)

    arc_losses = []
    label_losses = []

    for epoch in range(epochs):

        epoch_arc_loss = 0
        epoch_label_loss = 0

        if verbose > 0: print('\n***** Epoch {}:'.format(epoch))

        for sentence in sentences:

            sentence_var = Variable(embed_sentence(sentence, language), requires_grad=False)

            gold_arcs = Variable(calc_gold_arcs(sentence))
            gold_labels = Variable(calc_gold_labels(sentence))
            loss, arc_matrix, losses_separate = train_step(model, sentence_var, gold_arcs, gold_labels, arc_loss_criterion, label_loss_criterion, optimizer)
            epoch_arc_loss += losses_separate[0]
            epoch_label_loss += losses_separate[1]

            if verbose > 2: print('loss {0:.4f} for "'.format(loss.data.numpy()[0]) + ' '.join(word for word in sentence[:,0]) + '"')

            if verbose > 1: visualise_sentence(sentence, arc_matrix, epoch, modelname, language)

            del arc_matrix

        torch.save(model.state_dict(), "lang_{}/models/{}.pth".format(language, modelname))
        arc_losses.append(epoch_arc_loss.data.numpy()[0] / len(sentences))
        label_losses.append(epoch_label_loss.data.numpy()[0] / len(sentences))


        if verbose > 0: print('combined loss {} \n*****'.format(arc_losses[-1] + label_losses[-1]))

        pyplot.plot(range(len(arc_losses)), arc_losses, label='arc loss')
        pyplot.plot(range(len(label_losses)), label_losses, label='label loss')
        pyplot.legend(loc='upper right')
        pyplot.savefig('lang_{}/models/{}_loss.pdf'.format(language, modelname))
        print('arc loss: ', arc_losses[-1])
        print('label loss: ', label_losses[-1])


# Hyperparameters
lstm_in_size = 125
lstm_h_size = 125
lstm_num_layers = 1

MLP_in = 500

MLP_score_hidden = int(sys.argv[1])
MLP_score_out = 1

MLP_label_hidden = int(sys.argv[3])
MLP_label_out = 50

learning_rate = float(sys.argv[2])
arc_loss_criterion = nn.CrossEntropyLoss()
label_loss_criterion = nn.CrossEntropyLoss()
model = LSTMParser()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


language = sys.argv[4]
file = 'lang_{}/gold/{}-ud-train.conllu'.format(language, language)

if __name__ == "__main__":

    epochs = int(sys.argv[5])
    modelname = sys.argv[6]

    train(file, model, language, epochs)
