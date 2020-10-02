import numpy as np


def load_glove(preprocesser, glove_filename):
    word_to_embedding = {}

    with open(glove_filename, 'r', encoding='utf-8') as glove_file:
        for (i, line) in enumerate(glove_file):
            split = line.split(' ')

            word = split[0]
            embedding = split[1:]
            embedding = np.array([float(val) for val in embedding])
            
            if preprocesser.get_id(word) != '[UNL]':
                word_to_embedding[preprocesser.get_id(word)] = embedding
    return word_to_embedding
