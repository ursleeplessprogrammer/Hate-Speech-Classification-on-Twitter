"""Custom Keras metrics (program sections 4.16–4.17)."""
import tensorflow.keras.backend as K


def micro_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def micro_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def micro_f1_score(y_true, y_pred):
    precision = micro_precision(y_true, y_pred)
    recall = micro_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


def hamming_loss_metric(y_true, y_pred):
    y_pred = K.round(y_pred)
    return K.mean(K.not_equal(y_true, y_pred), axis=-1)
