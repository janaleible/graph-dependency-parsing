import numpy as np

word_embeddings = {
    'en': None,
    'de': None
}

tag_embeddings = {
    'en': None,
    'de': None
}

def concatenate(word_vec, tag_vec):
    con = np.concatenate((word_vec, tag_vec))
    return con

def get_word_embeddings(language) -> {}:

    if word_embeddings[language] is not None: return word_embeddings[language]

    embed_word = {}
    filename = 'lang_{}/embeddings/vectors-unk.txt'.format(language)
    with open(filename, "r") as f:
        for line in f:
            tokens = line.lower().strip().split(" ")
            vec = tokens
            x = vec.pop(0)
            embed_word[x] = [float(x) for x in vec]

    word_embeddings[language] = embed_word

    return embed_word

def get_tag_embeddings(language) -> {}:

    if tag_embeddings[language] is not None: return tag_embeddings[language]

    embed_tag = {}
    filename = 'lang_{}/embeddings/vectors-tags.txt'.format(language)
    with open(filename, "r") as f:
        for line in f:
            tokens = line.lower().strip().split(" ")
            vec = tokens
            x = vec.pop(0)
            embed_tag[x] = [float(x) for x in vec]

    tag_embeddings[language] = embed_tag

    return embed_tag