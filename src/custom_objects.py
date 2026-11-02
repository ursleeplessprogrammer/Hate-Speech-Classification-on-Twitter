"""Custom Keras objects required when loading a saved model (section 4.20)."""
from .attention import SeqAttentionLayer
from .metrics import (micro_precision, micro_recall, micro_f1_score,
                      hamming_loss_metric)

CUSTOM_OBJECTS = {
    "SeqAttentionLayer": SeqAttentionLayer,
    "micro_precision": micro_precision,
    "micro_recall": micro_recall,
    "micro_f1_score": micro_f1_score,
    "hamming_loss_metric": hamming_loss_metric,
}
