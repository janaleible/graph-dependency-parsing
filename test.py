import numpy as np
import NLP_training
import torch
from torch.autograd import Variable

language = 'en'
filename = 'lang_{}/gold/{}-ud-dev.conllu'.format(language, language)

sentences = NLP_training.prepare_data(filename)

model = NLP_training.LSTMParser()
model.load_state_dict(torch.load("lang_{}/models/model4.pth".format(language)))

def UAS_score(model, sentences):

    precision_arr = np.zeros(len(sentences))

    for i in range(len(sentences)):
        print(i)
        sentence = sentences[i]
        target = NLP_training.calc_gold_arcs(sentence)
        sentence_var = Variable(NLP_training.embed_sentence(sentence, language), requires_grad=False)

        max_tree, _ = model.predict(sentence_var)

        # prediction, _ = model(sentence_var)

        # softmax = nn.Softmax()
        # prediction = softmax(prediction)
        #
        # prediction = prediction.data.numpy()
        #
        # # we represent the final parse as a words*words mtx, where the root is indicated as the diagonal element
        # max_tree = mst(prediction)

        precision_sent = 0
        for j in range(len(sentence)):
            if(max_tree[j, target.view(target.size()[1])[j]] != 0):
                precision_sent += 1
            precision_arr[i] = precision_sent/(len(sentence))
    return np.mean(precision_arr)

# def write_to_file(filename, sentence, tree):
#
#     with open(filename, 'a') as file:
#



UAS = UAS_score(model, sentences)
print(UAS)