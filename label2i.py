from collections import defaultdict

import pickle

label2i = defaultdict(lambda: len(label2i))
i2label = defaultdict(lambda: len(i2label))

with open('lang_de/embeddings/vectors-labels.txt', 'r') as file:
    for line in file.readlines():
        label = line.split(' ')[0].lower()
        i2label[label2i[label]] = label

with open('lang_de/embeddings/label2i.pickle', 'wb') as file:
    pickle.dump(dict(label2i), file)

with open('lang_de/embeddings/i2label.pickle', 'wb') as file:
    pickle.dump(dict(i2label), file)