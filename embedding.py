import numpy as np

language = 'en'

def concatenate(word_vec, tag_vec):
    con = np.concatenate((word_vec, tag_vec))
    return con


embed_word = {}
filename = 'lang_{}/embeddings/vectors-unk.txt'.format(language)
with open(filename, "r") as f:
    for line in f:
        tokens = line.lower().strip().split(" ")
        vec = tokens
        x = vec.pop(0)
        embed_word[x] = [float(x) for x in vec]

embed_tag = {}
filename = 'lang_{}/embeddings/vectors-tags.txt'.format(language)
with open(filename, "r") as f:
    for line in f:
        tokens = line.lower().strip().split(" ")
        vec = tokens
        x = vec.pop(0)
        embed_tag[x] = [float(x) for x in vec]
