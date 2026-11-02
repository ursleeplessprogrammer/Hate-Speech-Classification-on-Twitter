"""Training metric logger (program section 4.19)."""
import pandas as pd
from tensorflow.keras.callbacks import Callback


class MetricsLogger(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_metrics = []
        self.epoch_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_metrics.append({
            "micro_precision": logs.get("micro_precision"),
            "micro_recall": logs.get("micro_recall"),
            "micro_f1_score": logs.get("micro_f1_score"),
            "hamming_loss": logs.get("hamming_loss_metric"),
        })
        self.epoch_accuracy.append({key: logs.get(key) for key in
            ("accuracy", "val_accuracy", "loss", "val_loss")})
        print("\nMicro Metrics Table:")
        print(pd.DataFrame(self.epoch_metrics))
        print("\nAccuracy Table:")
        print(pd.DataFrame(self.epoch_accuracy))
