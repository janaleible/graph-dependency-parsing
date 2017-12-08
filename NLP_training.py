import numpy as np
import embedding
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import calc_goldtree

filename = 'UD_english_reduced_training.txt'

def prepare_data(filename):

    properties = ["words", "tags", "dep_heads", "labels"]
    for i in range(len(properties)):
        properties[i] = []
    with open(filename, "r") as f:
        for line in f:
            tokens = line.lower().strip().split(" ")
            if(len(tokens)>4):
                continue
            else:
                for i in range(len(properties)):
                    properties[i].append(tokens[i].replace("mehhmeh",'","').strip(','))

    words, tags, dep_heads, labels = properties
    words.pop(0)
    tags.pop(0)
    dep_heads.pop(0)
    labels.pop(0)


    # make a list of numpy 2D arrays from all the sentences
    sentences = []
    end_signs = [".", "?", "!"]
    j = 0 #keeps track whether we are on the beginning of a sentence
    sent_idx = 0
    for i in range(len(words)):
        if (j == 0):
            sentences.append(np.array([words[i], tags[i], dep_heads[i], labels[i]]))
            j += 1
        elif (words[i] in end_signs):
            sentences[sent_idx] = np.vstack((sentences[sent_idx],np.array([words[i], tags[i], dep_heads[i], labels[i]])))
            sent_idx += 1
            j = 0
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
            try:
                sentence_array[i, :] = embedding.concatenate(embedding.embed_word["<unk>"], embedding.embed_tag[s[i, 1]])
            except(KeyError):
                pass

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
            heads[i]=int(i)
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
    print(loss)
    loss.backward()
    optimizer.step()

    return output_mtx, loss


# define the training for loop
sentences = prepare_data(filename)

for epoch in range(1):
    for i in range(len(sentences)):

        # Embed sentences[i]
        s = sentences[i]
        sentence_var = Variable(embed_sentence(s), requires_grad=False)

        try:
            gold = Variable(calc_gold(s))
            parse_mtx, loss = train_step(model, sentence_var, gold, loss_criterion, optimizer)
        except(RuntimeError or ValueError or KeyError):
            pass











