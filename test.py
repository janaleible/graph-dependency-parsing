import numpy as np
import NLP_training
import os
import torch
from torch.autograd import Variable
import torch.nn as nn

filename = 'UD_english_reduced_test.txt'

sentences = NLP_training.prepare_data(filename)

model = NLP_training.LSTMParser()
model.load_state_dict(torch.load(os.getcwd()+"/my_model.pth"))

def UAS_score(model, sentences):
    precision_arr = np.zeros(len(sentences))
    for i in range(len(sentences)):
        s = sentences[i]
        target = NLP_training.calc_gold(s)
        s_var = Variable(NLP_training.embed_sentence(s), requires_grad=False)
        predict =model(s_var)

        # originally predict(i,j) means the number on the arc from i to j, so first we transpose, then softmax over the rows, then transpose back (because Edmonds is in that format)

        # Todo Edmonds DOES NOT work on softmax, right??
        # predict = torch.transpose((predict), 0, 1)
        # softm = nn.Softmax()
        # predict = softm(predict)
        # predict = torch.transpose((predict), 0, 1)
        
        # converting it to numpy because of the Edmonds algorithm. Now the rows are the
        predict = predict.numpy()


        # To apply Edmonds algorithm, we have to convert the matrix to the desired format: Root is not

        precision_sent = 0
        for j in range(len(s)):
            if(predict[j,target.view(target.size()[1])[j]] == 1):
                precision_sent += 1
            precision_arr[i] = precision_sent/len(s)
    return np.mean(precision_arr)


UAS = UAS_score(model, sentences)
print(UAS)