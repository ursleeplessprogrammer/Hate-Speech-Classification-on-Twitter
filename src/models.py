"""SGM model builders and training helper (program sections 4.21–4.27).

The source describes binary, five-output multi-label, and three-class models.
The GRU variants in sections 4.25–4.27 append a GRU after the LSTM decoder.
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Embedding, Bidirectional, LSTM, Dense,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout, GRU)
from tensorflow.keras.callbacks import ModelCheckpoint
from .attention import SeqAttentionLayer
from .metrics import (micro_precision, micro_recall, micro_f1_score,
                      hamming_loss_metric)
from .callbacks import MetricsLogger


def build_sgm_model(vocab_size, embedding_dim, max_sequence_length,
                    embeddings_matrix, task="binary", use_gru=False,
                    encoder_units=None, decoder_units=None, gru_units=64,
                    dropout_rate=0.3):
    """Build an SGM classifier using the architectures described in the PDF.

    task: 'binary' (1 sigmoid), 'multilabel' (5 softmax outputs), or
    'multiclass' (3 softmax outputs). Defaults reflect the source architectures.
    """
    configs = {
        "binary": (1, "sigmoid", "binary_crossentropy", 32, 32, False),
        "multilabel": (5, "softmax", "categorical_crossentropy", 128, 128, True),
        "multiclass": (3, "softmax", "categorical_crossentropy", 128, 128, True),
    }
    if task not in configs:
        raise ValueError("task must be 'binary', 'multilabel', or 'multiclass'")
    outputs, activation, loss, default_encoder, default_decoder, use_max_pool = configs[task]
    encoder_units = encoder_units or default_encoder
    decoder_units = decoder_units or default_decoder

    model = Sequential(name=f"sgm_{task}")
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                        input_length=max_sequence_length,
                        weights=[embeddings_matrix], trainable=True))
    model.add(Bidirectional(LSTM(units=encoder_units, return_sequences=True)))
    model.add(SeqAttentionLayer())
    model.add(LSTM(units=decoder_units, return_sequences=True))
    if use_gru:
        model.add(GRU(gru_units, return_sequences=True))
    model.add(GlobalMaxPooling1D() if use_max_pool else GlobalAveragePooling1D())
    if task != "binary" or use_gru:
        model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(outputs, activation=activation))
    model.compile(optimizer="adam", loss=loss,
                  metrics=["accuracy", micro_precision, micro_recall,
                           micro_f1_score, hamming_loss_metric])
    return model


def train_model(model, X_train, y_train, output_path="SGM_Model.keras",
                epochs=5, batch_size=64, validation_split=0.1):
    checkpoint = ModelCheckpoint(output_path, monitor="val_accuracy",
                                 verbose=1, save_best_only=True, mode="max")
    logger = MetricsLogger()
    history = model.fit(X_train, y_train, validation_split=validation_split,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[checkpoint, logger])
    return history


def save_model(model, output_path):
    model.save(output_path)


def load_model(model_path, custom_objects=None):
    custom_objects = custom_objects or {}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
