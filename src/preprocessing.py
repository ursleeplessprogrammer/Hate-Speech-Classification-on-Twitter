"""Text cleaning and preprocessing functions (program sections 4.3–4.6)."""
import codecs
import re
from nltk import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

_stemmer = StemmerFactory().create_stemmer()


def lowercase(text):
    return str(text).lower()


def remove_non_ascii(text):
    try:
        text = codecs.decode(text.encode("utf-8"), "unicode-escape")
    except UnicodeDecodeError:
        text = text.encode("utf-8").decode("unicode-escape", errors="ignore")
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return re.sub(r"[\n\t\s]+", " ", text).strip()


def remove_unnecessary_char(text):
    text = remove_non_ascii(text.lower())
    text = re.sub(r"rt", " ", text)
    text = re.sub(r"user", " ", text)
    text = re.sub(r"pengguna", " ", text)
    text = re.sub(r"((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))", " ", text)
    return re.sub(r"  +", " ", text).strip()


def remove_non_alphanumeric(text):
    return re.sub(r"[^0-9a-zA-Z]+", " ", text)


def normalize_alay(text, slang_mapping):
    """Replace tokens using a dict: {informal_word: normalized_word}."""
    return " ".join(slang_mapping.get(word, word) for word in text.split(" "))


def remove_stopwords(text, stopwords):
    stopword_set = set(stopwords)
    return re.sub(r"  +", " ", " ".join(
        "" if word in stopword_set else word for word in text.split(" ")
    )).strip()


def stemming(text, stemmer=None):
    return (stemmer or _stemmer).stem(text)


def form_sentence(text):
    return word_tokenize(text)


def preprocess(text, slang_mapping=None, stopwords=None, stemmer=None):
    """Apply the source pipeline: lowercase, clean, normalize, stem, remove stopwords.

    Pass slang_mapping and stopwords loaded from data_preparation.py.
    """
    slang_mapping = slang_mapping or {}
    stopwords = stopwords or []
    text = lowercase(text)
    text = remove_unnecessary_char(text)
    text = remove_non_alphanumeric(text)
    text = normalize_alay(text, slang_mapping)
    text = stemming(text, stemmer)
    return remove_stopwords(text, stopwords)
