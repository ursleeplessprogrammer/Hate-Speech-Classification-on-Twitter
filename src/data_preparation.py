"""Dataset loading, reference dictionaries, and train/test splitting.
Based on program sections 4.1, 4.4, 4.5 and 4.8.
"""
from pathlib import Path
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv_detect_encoding(file_path, **kwargs):
    """Read a CSV after detecting its file encoding (source sections 4.1/4.4/4.5)."""
    with open(file_path, "rb") as file:
        encoding = chardet.detect(file.read()).get("encoding") or "utf-8"
    return pd.read_csv(file_path, encoding=encoding, **kwargs)


def load_original_dataset(file_path, tweet_column="Tweet"):
    """Load the original CSV whose first row contains its column names."""
    df = read_csv_detect_encoding(file_path, header=None)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    if tweet_column in df.columns:
        df[tweet_column] = df[tweet_column].fillna("").astype(str)
    return df


def load_stopwords(file_path):
    """Load the stopword list used by the thesis preprocessing pipeline."""
    df = read_csv_detect_encoding(file_path, header=None)
    return df.rename(columns={0: "stopword"})


def load_slang_dictionary(file_path):
    """Load informal-to-formal word mapping (kamusalay)."""
    df = read_csv_detect_encoding(file_path)
    return df.rename(columns={0: "original", 1: "replacement"})


def split_dataset(data, labels, test_size=0.3, random_state=42):
    """Split text and labels into train and test sets, following section 4.8."""
    return train_test_split(data, labels, test_size=test_size,
                            random_state=random_state)
