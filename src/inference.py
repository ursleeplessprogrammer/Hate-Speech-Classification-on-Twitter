"""Model loading and hierarchical prediction pipeline (sections 4.30–4.35)."""
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .preprocessing import preprocess


def classify_with_model(model, tokenizer, text, preprocess_kwargs=None,
                        max_sequence_length=256, threshold=0.5):
    cleaned = preprocess(text, **(preprocess_kwargs or {}))
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_sequence_length,
                           padding="post", truncating="post")
    prediction = model.predict(padded, verbose=0)
    return int(float(np.asarray(prediction).reshape(-1)[0]) >= threshold)


def classify_with_model_4(text, models, tokenizers, preprocess_kwargs=None,
                          max_sequence_length=256, threshold=0.5,
                          other_threshold=0.3):
    """Predict five Model-4 labels; models/tokenizers ordered by label."""
    labels = ["HS_Religion", "HS_Race", "HS_Physical", "HS_Gender", "HS_Other"]
    if len(models) != 5 or len(tokenizers) != 5:
        raise ValueError("Model 4 requires five models and five tokenizers")
    cleaned = preprocess(text, **(preprocess_kwargs or {}))
    predictions = []
    for model, tokenizer in zip(models, tokenizers):
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_sequence_length,
                               padding="post", truncating="post")
        predictions.append(float(np.asarray(model.predict(padded, verbose=0)).reshape(-1)[0]))
    values = [int(score >= threshold) for score in predictions[:4]]
    other = 0 if any(values) else int(predictions[4] >= other_threshold)
    return dict(zip(labels, values + [other]))


def classify_with_model_5(model, tokenizer, text, preprocess_kwargs=None,
                          max_sequence_length=256):
    cleaned = preprocess(text, **(preprocess_kwargs or {}))
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_sequence_length,
                           padding="post", truncating="post")
    prediction = model.predict(padded, verbose=0)
    labels = ["HS_Weak", "HS_Moderate", "HS_Strong"]
    return labels[int(np.argmax(prediction, axis=1)[0])]


def classify_pipeline(text, models, tokenizers, preprocess_kwargs=None,
                      max_sequence_length=256, threshold=0.5):
    """Run Models 1–5 following the conditional flow in program section 4.32.

    models keys: model_1, model_2, model_3, model_4 (five-model list), model_5.
    tokenizers keys: tokenizer_1, tokenizer_2, tokenizer_3,
    tokenizer_4 (five-tokenizer list), tokenizer_5.
    """
    m1 = classify_with_model(models["model_1"], tokenizers["tokenizer_1"], text,
        preprocess_kwargs, max_sequence_length, threshold)
    results = {"Model_1": "Hate-Speech" if m1 else "Not Hate-Speech",
               "Model_2": 0, "Model_3": 0, "Model_4": 0, "Model_5": 0}
    m2 = classify_with_model(models["model_2"], tokenizers["tokenizer_2"], text,
        preprocess_kwargs, max_sequence_length, threshold)
    results["Model_2"] = "Abusive" if m2 else "Not Abusive"
    if m1:
        m3 = classify_with_model(models["model_3"], tokenizers["tokenizer_3"], text,
            preprocess_kwargs, max_sequence_length, threshold)
        results["Model_3"] = "Target: Individual" if m3 else "Target: Group"
        m4 = classify_with_model_4(text, models["model_4"], tokenizers["tokenizer_4"],
            preprocess_kwargs, max_sequence_length, threshold)
        results["Model_4"] = [label for label, value in m4.items() if value]
        results["Model_5"] = classify_with_model_5(models["model_5"],
            tokenizers["tokenizer_5"], text, preprocess_kwargs, max_sequence_length)
    return results
