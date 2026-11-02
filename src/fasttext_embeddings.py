"""FastText loading and embedding-matrix creation (sections 4.13–4.15)."""
import numpy as np
import fasttext


def load_fasttext_model(model_path):
    return fasttext.load_model(model_path)


def build_fasttext_embedding_matrix(tokenizer, fasttext_model, embedding_dim=300):
    matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, index in tokenizer.word_index.items():
        vector = fasttext_model.get_word_vector(word)
        if vector is not None:
            matrix[index] = vector
    return matrix
