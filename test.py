import numpy as np
import NLP_training
import torch
from torch.autograd import Variable

language = 'en'
filename = 'lang_{}/gold/{}-ud-test.conllu'.format(language, language)
# filename = 'lang_en/gold/mytest.conllu'

sentences = NLP_training.prepare_data(filename, training=False)

model = NLP_training.LSTMParser()
model.load_state_dict(torch.load("lang_{}/models/model11.pth".format(language)))

def UAS_score(model, sentences):

    precision_arr = np.zeros(len(sentences))

    with open('conllu', 'w') as file:
        file.write('')

    precision_sent = 0
    word_count = 0
    for i in range(len(sentences)):
        print(i)
        sentence = sentences[i]
        target = NLP_training.calc_gold_arcs(sentence)
        sentence_var = Variable(NLP_training.embed_sentence(sentence, language), requires_grad=False)

        max_tree, _ = model.predict(sentence_var)

        # precision_sent = 0
        for j in range(1, len(sentence)):
            if(max_tree[j, target.view(target.size()[1])[j]] != 0):
                precision_sent += 1
            word_count += 1
            # precision_arr[i] = precision_sent/(len(sentence) - 1)

        write_to_file('conllu', sentence[1:], max_tree[1:,])

    # return np.mean(precision_arr)
    return precision_sent / word_count

def write_to_file(filename, sentence, tree):

    conllu = '# ' + ' '.join(sentence[:, 0]) + '\n'
    for index, line in enumerate(sentence):
        conllu += '{}\t{}\t_\t_\t_\t_\t{}\t_\t_\t_'.format(index + 1, line[0], np.argmax(tree[index, :]), )
        conllu += '\n'
    conllu += '\n'

    with open(filename, 'a') as file:
        file.write(conllu)




UAS = UAS_score(model, sentences)
print(UAS)