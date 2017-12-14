import numpy as np
import NLP_training
import torch
from torch.autograd import Variable
import Edmonds_m
import torch.nn as nn

from mst import mst

language = 'en'
filename = 'lang_{}/gold/{}-ud-test.conllu'.format(language, language)

sentences = NLP_training.prepare_data(filename)

model = NLP_training.LSTMParser()
model.load_state_dict(torch.load("lang_{}/models/my_model.pth".format(language)))

def UAS_score(model, sentences):

    precision_arr = np.zeros(len(sentences))

    for i in range(len(sentences)):
        print(i)
        sentence = sentences[i]
        target = NLP_training.calc_gold(sentence)
        sentence_var = Variable(NLP_training.embed_sentence(sentence, language), requires_grad=False)
        prediction = model(sentence_var)

        # prediction = torch.transpose((prediction), 0, 1)
        softmax = nn.Softmax()
        prediction = softmax(prediction)
        # prediction = torch.transpose((prediction), 0, 1)

        # converting it to numpy because of the Edmonds algorithm. Now the rows are the
        prediction = prediction.data.numpy()


        # To apply Edmonds algorithm, we have to convert the matrix to the desired format: root is a vector and not represented in the matrix
        # root = np.diag(prediction)
        # A = prediction
        # span_value = np.zeros(len(root))
        # for j in range(len(root)):
        #     span_value[j] = sum(sum(Edmonds_m.Edmonds(A, j)))
        #
        # max_parse_values = span_value + root
        # root_ind = np.argmax(max_parse_values)

        # we represent the final parse as a words*words mtx, where the root is indicated as the diagonal element
        max_tree = mst(prediction)
        # max_tree[root_ind, root_ind] = root[root_ind]

        precision_sent = 0
        for j in range(len(sentence)):
            if(max_tree[j,target.view(target.size()[1])[j]] != 0):
                precision_sent += 1
            precision_arr[i] = precision_sent/(len(sentence))
    return np.mean(precision_arr)


UAS = UAS_score(model, sentences)
print(UAS)