import numpy as np


def load_glove(voc_file, glove_filename, voc_size):
    word_to_id = {}
    word_to_embedding = {}

    vocab_dict = open(voc_file, encoding='utf-8').readlines()
    for line in vocab_dict[:voc_size]:
        word, word_id = line.split()
        word_to_id[word] = word_id

    with open(glove_filename, 'r') as glove_file:
        for (i, line) in enumerate(glove_file):
            split = line.split(' ')

            word = split[0]
            embedding = split[1:]
            embedding = np.array([float(val) for val in embedding])
            
            if word in word_to_id.keys():
                word_to_embedding[int(word_to_id[word])] = embedding
    return word_to_embedding
