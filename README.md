# Hate-Speech-Classification-on-Twitter

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/IDE-Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
</p>

**Multi-Label Classification using Sequential Generation Model with Gated Recurrent Unit and FastText on Hate-Speech Text on the Twitter Platform.**

This repository contains the implementation of a deep learning-based system designed to classify Indonesian hate speech on Twitter into multiple labels. The project utilizes a Sequential Generation Model (SGM) architecture, enhanced with Gated Recurrent Units (GRU) and evaluated using GLoVe and FastText word embeddings.

---

## 📌 Project Overview

Hate speech on social media is a complex issue where a single post often contains multiple categories of abuse (e.g., targeting both religion and race). This project addresses the Multi-Label Classification challenge by treating label prediction as a sequence generation task.

### Key Features

- **Architecture:** Sequential Generation Model (SGM).

  <img width="650" height="404" alt="SGM Architecture" src="https://github.com/user-attachments/assets/5b3312a2-c0eb-4e21-b210-9bd93f4b1fc7" />

- **Encoder:** Bidirectional Long Short-Term Memory (Bi-LSTM).
- **Decoder:** LSTM integrated with Gated Recurrent Units (GRU).

  <img width="483" height="293" alt="Model Architecture" src="https://github.com/user-attachments/assets/ceebbd68-da05-4ef3-83da-51584b45154f" />

- **Mechanism:** Attention Mechanism to focus on relevant keywords.
- **Embeddings:** Comparative analysis between GLoVe and FastText.
- **Data Augmentation:** Implementation of Back-Translation to balance the dataset.
- **Deployment:** Web-based interface built with Flask.

---

## 🏗️ System Architecture

The model is divided into five specialized sub-models to handle specific classification tasks:

| Model | Classification Task | Description |
|---|---|---|
| Model A | Hate Speech | Binary classification (HS vs. Non-HS). |
| Model B | Abusive | Binary classification (Abusive vs. Non-Abusive). |
| Model C | HS Target | Classification of target (Individual vs. Group). |
| Model D | HS Category | Multi-label classification (Religion, Race, Physical, Gender, Other). |
| Model E | HS Level | Multi-class classification of severity (Weak, Moderate, Strong). |

---

## 📊 Dataset

The research uses the Indonesian Twitter Hate Speech dataset (Ibrohim et al., 2019).

| Dataset Information | Details |
|---|---|
| Original Data | 13,169 tweets |
| Augmented Data | 149,824 tweets |
| Augmentation Method | Back-Translation |
| Classification Labels | 12 distinct labels |

Back-Translation was implemented to address label imbalance and increase the size of the training dataset.

---

## 📁 Project Structure

The source code is organized into separate Python modules, with each module responsible for a specific part of the machine learning pipeline.

| File | Responsibility | Source Sections |
|---|---|---|
| `data_preparation.py` | CSV loading, stopwords/slang dictionaries, dataset splitting | Data preparation |
| `augmentation.py` | Back-translation | Data augmentation |
| `preprocessing.py` | Text cleaning, normalization, stemming, stopword removal | Text preprocessing |
| `tokenization.py` | Tokenizer training, save/load, and sequence padding | Tokenization |
| `glove_embeddings.py` | Word2Vec fine-tuning, GLoVe conversion, and embedding matrix generation | GLoVe embeddings |
| `fasttext_embeddings.py` | FastText loading and embedding matrix generation | FastText embeddings |
| `metrics.py` | Micro precision, recall, F1-score, and Hamming loss | Evaluation metrics |
| `attention.py` | Sequence attention layer | Attention mechanism |
| `callbacks.py` | Training metrics logger | Model training |
| `custom_objects.py` | Keras custom object registry | Model configuration |
| `models.py` | SGM model building, training, saving, and loading | Model architecture |
| `visualization.py` | Training-history plots | Visualization |
| `inference.py` | Model-specific and hierarchical prediction functions | Prediction |
| `flask_app.py` | Flask prediction API | Deployment |

---

## 🚀 Performance Results

The experiments concluded that the GLoVe embedding combined with a GRU layer in the decoder yielded the highest performance.

### Best Model: SGM + GRU

| Evaluation Metric | Score |
|---|---:|
| Hamming Loss | 0.005481 |
| Micro-F1 Score | 0.993527 |
| Accuracy | 0.994353 |

**Key Finding:** The addition of a GRU layer significantly improved the model's ability to capture dependencies between labels compared to the standard SGM.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python 3.10.12 |
| Deep Learning | TensorFlow, Keras |
| NLP Libraries | NLTK, Sastrawi |
| Word Embeddings | Gensim (Word2Vec/GLoVe), FastText |
| Data Processing | Pandas, NumPy |
| Machine Learning Utilities | Scikit-learn |
| Web Framework | Flask |
| Translation API | Googletrans / Deep Translator |
| Visualization | Matplotlib |

---

## 💻 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/ursleeplessprogrammer/Hate-Speech-Classification-On-Twitter.git

cd Hate-Speech-Classification-On-Twitter
```

### 2. Install Dependencies

Install the packages used by the source project in your Python environment.

```bash
pip install pandas numpy chardet scikit-learn deep-translator nltk Sastrawi gensim fasttext tensorflow matplotlib flask
```

### 3. Download NLTK Data

Download the NLTK tokenizer data once before running the preprocessing pipeline.

```python
import nltk

nltk.download("punkt")
```

### 4. Run the Flask Application

```bash
python flask_app.py
```

Access the web interface through your browser:

```text
http://127.0.0.1:5000
```

---

## 🧪 Basic Usage

The project is modularized, allowing individual components of the machine learning pipeline to be imported and used independently.

For example, the following code demonstrates how to load the original dataset, stopwords, and slang dictionary before preprocessing Indonesian text.

```python
from project_hate_speech_c14200138.data_preparation import (
    load_original_dataset,
    load_stopwords,
    load_slang_dictionary,
    split_dataset
)

from project_hate_speech_c14200138.preprocessing import preprocess

# Update these paths to match your local files.
df = load_original_dataset("/data/re_dataset.csv")

stopword_df = load_stopwords("/data/stopwordbahasa.csv")

slang_df = load_slang_dictionary("/data/new_kamusalay.csv")

# Prepare stopwords and slang mapping.
stopwords = (
    stopword_df["stopword"]
    .dropna()
    .astype(str)
    .tolist()
)

slang_mapping = dict(
    zip(
        slang_df["original"],
        slang_df["replacement"]
    )
)

# Preprocess an Indonesian text sample.
cleaned_text = preprocess(
    "Contoh teks...",
    slang_mapping,
    stopwords
)
```

**Note:** Update dataset paths and package import paths according to your local project directory structure.

---

## 📝 Preprocessing Workflow

The system performs several preprocessing operations on raw Indonesian text before passing it to the classification model.

| Step | Process | Description |
|---|---|---|
| 1 | Case Folding | Converts text into lowercase. |
| 2 | Cleaning | Removes URLs, usernames (@user), and special characters. |
| 3 | Emoji Handling | Converts emojis into text representations. |
| 4 | Normalization | Converts informal Indonesian (slang) into formal language using a custom lexicon. |
| 5 | Stopword Removal | Filters out non-essential words. |
| 6 | Stemming | Reduces words to their base form using the Sastrawi library. |

---

## 📚 References

- Yang, P., et al. (2018). SGM: Sequence Generation Model for Multi-Label Classification.
- Ibrohim, M. O., & Budi, I. (2019). Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter.
- Asrawi, H., et al. (2023). LSTM and Bidirectional GRU Comparison for Text Classification.

---
## 📄 License

Please refer to the repository for applicable licensing information.

## 👨‍💻 Author

**ursleeplessprogrammer**
