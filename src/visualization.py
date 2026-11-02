"""Training history plots (program section 4.29)."""
import matplotlib.pyplot as plt


def plot_training_history(history):
    history_dict = history.history
    epochs = range(1, len(history_dict["accuracy"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(epochs, history_dict["accuracy"], label="Training accuracy")
    axes[0, 0].plot(epochs, history_dict["val_accuracy"], label="Validation accuracy")
    axes[0, 0].set(title="Training and validation accuracy", xlabel="Epochs", ylabel="Accuracy")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, history_dict["loss"], label="Training loss")
    axes[0, 1].plot(epochs, history_dict["val_loss"], label="Validation loss")
    axes[0, 1].set(title="Training and validation loss", xlabel="Epochs", ylabel="Loss")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, history_dict["micro_precision"], label="Micro Precision")
    axes[1, 0].set(title="Micro-averaged precision", xlabel="Epochs", ylabel="Precision")
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, history_dict["micro_recall"], label="Micro Recall")
    axes[1, 1].plot(epochs, history_dict["micro_f1_score"], label="Micro F1-score")
    axes[1, 1].set(title="Micro-averaged recall and F1-score", xlabel="Epochs", ylabel="Score")
    axes[1, 1].legend()
    fig.tight_layout()
    return fig
