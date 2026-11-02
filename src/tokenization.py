"""Tokenizer persistence and sequence padding (program sections 4.7–4.8)."""
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def train_tokenizer(texts, num_words=None, **kwargs):
    tokenizer = Tokenizer(num_words=num_words, **kwargs)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_tokenizer(tokenizer, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def texts_to_padded_sequences(texts, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_sequence_length,
                         padding="post", truncating="post")
