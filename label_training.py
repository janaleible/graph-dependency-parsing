import sys

import numpy as np
import torch
import pickle

from matplotlib import pyplot
from torch.autograd import Variable

from NLP_training import LSTMParser, prepare_data, embed_sentence


class LabelMLP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # self.weights = Parameter()

        self.labelMLP = torch.nn.Sequential(
            torch.nn.Linear(MLP_in, MLP_label_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(MLP_label_hidden, MLP_label_out)
        )

    def forward(self, biLSTM_sentence, tree):

        heads = biLSTM_sentence
        dependants = torch.index_select(
            biLSTM_sentence.data,
            0,
            torch.from_numpy(np.argmax(tree, 1))
        )

        concatenated = torch.cat((heads, dependants), 2).view(-1, 500)
        return self.labelMLP(concatenated)


def calc_gold_labels(sentence):

    labels = []

    with open('lang_en/embeddings/label2i.pickle', 'rb') as file:
        label2i = pickle.load(file)

    for line in sentence:
        labels.append(label2i[line[3]])

    target = torch.from_numpy(np.array([labels]))

    return target


# TRAINING
def train_step(label_model, input_sent, gold_labels, loss_criterion, optimizer, tree):

    label_model.zero_grad()
    labels_matrix = label_model(input_sent, tree)

    loss = loss_criterion(labels_matrix, gold_labels.view(gold_labels.size()[1]))
    loss.backward()
    optimizer.step()

    return loss


# define the training for loop
def train(filename, language, model, biLSTMModel, verbose = 1):

    sentences = prepare_data(filename)

    losses = []

    for epoch in range(int(sys.argv[3])):

        epoch_loss = 0

        if verbose > 0: print('\n***** Epoch {}:'.format(epoch))

        for sentence in sentences:

            sentence_var = Variable(embed_sentence(sentence, language), requires_grad=False)

            gold_labels = Variable(calc_gold_labels(sentence))

            tree, sentence_LSTM_encoding = biLSTMModel.predict(sentence_var)

            loss = train_step(model, Variable(sentence_LSTM_encoding.data), gold_labels, loss_criterion, optimizer, tree)

            epoch_loss += loss

            if verbose > 1: print('loss {0:.4f} for "'.format(loss.data.numpy()[0]) + ' '.join(word for word in sentence[:,0]) + '"')

        torch.save(model.state_dict(), "lang_{}/models/{}.pth".format(language, sys.argv[4]))
        losses.append(epoch_loss.data.numpy()[0] / len(sentences))

        if verbose > 0: print('average loss {} \n*****'.format(losses[-1]))

        pyplot.plot(range(len(losses)), losses)
        pyplot.savefig('lang_{}/models/{}_loss.pdf'.format(language, sys.argv[4]))


language = 'en'
filename = 'lang_{}/gold/{}-ud-train.conllu'.format(language, language)

paser_model = LSTMParser()
paser_model.load_state_dict(torch.load("lang_{}/models/model1.pth".format(language)))

MLP_in = 500

MLP_label_hidden = int(sys.argv[5])
MLP_label_out = 50

learning_rate = float(sys.argv[2])
loss_criterion = torch.nn.CrossEntropyLoss()
model = LabelMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    train(filename, language, model, paser_model)