"""Word2Vec fine-tuning and GLoVe-format helpers (sections 4.9–4.12)."""
import pickle
import numpy as np
from gensim.models import Word2Vec


def fine_tune_word2vec(model_path, sentences, output_path, epochs=15):
    model = Word2Vec.load(model_path)
    model.build_vocab(sentences, update=True)
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    model.save(output_path)
    return model


def save_word2vec_as_glove(model, glove_file_path):
    with open(glove_file_path, "w", encoding="utf-8") as file:
        for word in model.wv.index_to_key:
            vector_str = " ".join(map(str, model.wv[word]))
            file.write(f"{word} {vector_str}\n")


def load_glove_vectors(glove_file_path):
    embeddings = {}
    with open(glove_file_path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            if values:
                embeddings[values[0]] = list(map(float, values[1:]))
    return embeddings


def save_embeddings_pickle(data, pickle_file_path):
    with open(pickle_file_path, "wb") as file:
        pickle.dump(data, file)


def load_embeddings_pickle(pickle_file_path):
    with open(pickle_file_path, "rb") as file:
        return pickle.load(file)


def build_glove_embedding_matrix(tokenizer, embeddings_index):
    embedding_dim = len(next(iter(embeddings_index.values())))
    matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, index in tokenizer.word_index.items():
        vector = embeddings_index.get(word)
        if vector is not None:
            matrix[index] = vector
    return matrix
