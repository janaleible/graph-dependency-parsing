import numpy as np
import NLP_training
import torch
import pickle
from torch.autograd import Variable

language = 'en'
filename = 'lang_{}/gold/{}-ud-test.conllu'.format(language, language)
# filename = 'lang_en/gold/mytest.conllu'

sentences = NLP_training.prepare_data(filename, training=False)

model = NLP_training.LSTMParser()
model.load_state_dict(torch.load("lang_{}/models/model_att1.pth".format(language)))

def UAS_and_LAS(model, sentences, language):

    with open('conllu', 'w') as file:
        file.write('')

    with open('lang_{}/embeddings/i2label.pickle'.format(language), 'rb') as file:
        i2label = pickle.load(file)

    correct_arcs = 0
    correct_labels = 0
    word_count = 0

    for i in range(len(sentences)):

        sentence = sentences[i]

        target_arcs = NLP_training.calc_gold_arcs(sentence)
        target_labels = NLP_training.calc_gold_labels(sentence)
        target_labels_readable = [i2label[l] for l in list(target_labels.view(-1))]

        sentence_var = Variable(NLP_training.embed_sentence(sentence, language), requires_grad=False)

        arc_prediction, label_prediction = model.predict(sentence_var)

        for j, word in enumerate(sentence):

            if j == 0: continue # skip root

            word_count += 1

            if arc_prediction[j, target_arcs.view(-1)[j]] != 0 :
                correct_arcs += 1

            predicted_labels = np.argmax(label_prediction.data.numpy(), 1)

            if predicted_labels[j] == target_labels.view(-1)[j]:
                correct_labels += 1

        write_to_file('conllu', sentence[1:], arc_prediction[1:,], [i2label[l] for l in predicted_labels[1:]])

    return correct_arcs / word_count, correct_labels / word_count

def write_to_file(filename, sentence, tree, labels):

    conllu = '# ' + ' '.join(sentence[:, 0]) + '\n'

    for index, line in enumerate(sentence):
        conllu += '{}\t{}\t_\t_\t_\t_\t{}\t{}\t_\t_\n'.format(index + 1, line[0], np.argmax(tree[index, :]), labels[index])

    conllu += '\n'

    with open(filename, 'a') as file:
        file.write(conllu)




UAS = UAS_and_LAS(model, sentences,language)
print(UAS)