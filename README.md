# Hate-Speech-Classification-on-Twitter
Multi-Label Classification using Sequential Generation Model with Gated Recurrent Unit and FastText on Hate-Speech Text on the Twitter Platform.

This repository contains the implementation of a deep learning-based system designed to classify Indonesian hate speech on Twitter into multiple labels. The project utilizes a Sequential Generation Model (SGM) architecture, enhanced with Gated Recurrent Units (GRU) and evaluated using GLoVe and FastText word embeddings.



## 📌 Project Overview
Hate speech on social media is a complex issue where a single post often contains multiple categories of abuse (e.g., targeting both religion and race). This project addresses the Multi-Label Classification challenge by treating label prediction as a sequence generation task.

# Key Features:
- Architecture: Sequential Generation Model (SGM).
<img width="637" height="404" alt="image" src="https://github.com/user-attachments/assets/5b3312a2-c0eb-4e21-b210-9bd93f4b1fc7" />

- Encoder: Bidirectional Long Short-Term Memory (Bi-LSTM).

- Decoder: LSTM integrated with Gated Recurrent Units (GRU).
<img width="483" height="293" alt="image" src="https://github.com/user-attachments/assets/ceebbd68-da05-4ef3-83da-51584b45154f" />

- Mechanism: Attention Mechanism to focus on relevant keywords.

- Embeddings: Comparative analysis between GLoVe and FastText.

- Data Augmentation: Implementation of Back-Translation to balance the dataset.

- Deployment: Web-based interface built with Flask.



## 🏗️ System Architecture
The model is divided into 5 specialized sub-models to handle specific classification tasks:

- Model A (Hate Speech): Binary classification (HS vs. Non-HS).

- Model B (Abusive): Binary classification (Abusive vs. Non-Abusive).

- Model C (HS Target): Classification of target (Individual vs. Group).

- Model D (HS Category): Multi-label classification (Religion, Race, Physical, Gender, Other).

- Model E (HS Level): Multi-class classification of severity (Weak, Moderate, Strong).



## 📊 Dataset
The research uses the Indonesian Twitter Hate Speech dataset (Ibrohim et al., 2019).

Original Data: 13,169 tweets.

Augmented Data: 149,824 tweets (achieved via Back-Translation to resolve label imbalance).

Labels: 12 distinct labels covering HS types, targets, and severity levels.



## 🚀 Performance Results
The experiments concluded that the GLoVe embedding combined with a GRU layer in the decoder yielded the highest performance:

      Metric	         Average Score
      Hamming Loss	   0.005481
      Micro-F1 Score   	0.993527
      Accuracy	         0.994353

Key Finding: The addition of a GRU layer significantly improved the model's ability to capture dependencies between labels compared to the standard SGM.



## 🛠️ Tech Stack
- Language: Python 3.10.12

- Deep Learning: TensorFlow / Keras

- NLP Libraries: NLTK, Sastrawi (for Indonesian stemming)

- Word Embeddings: Gensim (Word2Vec/GLoVe), FastText

- Web Framework: Flask

- Translation API: Googletrans (for data augmentation)



## 💻 Installation & Usage
1. **Clone the repository**

   ```bash
   git clone https://github.com/ursleeplessprogrammer/Hate-Speech-Classification-On-Twitter.git
   cd Hate-Speech-Classification-On-Twitter
   
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt

3. **Run the Flask App**

   ```bash
   python app.py

Access the UI: Open http://127.0.0.1:5000 in your browser.



## 📝 Preprocessing Workflow
The system performs the following steps on raw Indonesian text:
- Case Folding: Converting text to lowercase.

- Cleaning: Removing URLs, usernames (@user), and special characters.

- Emoji Handling: Converting emojis into text representations.

- Normalization: Converting informal Indonesian (slang) to formal language using a custom lexicon.

- Stopword Removal: Filtering out non-essential words.

- Stemming: Reducing words to their base form using the Sastrawi library.



## 📚 References
- Yang, P., et al. (2018). SGM: Sequence Generation Model for Multi-Label Classification.
- Ibrohim, M. O., & Budi, I. (2019). Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter.
- Asrawi, H., et al. (2023). LSTM and Bidirectional GRU Comparison for Text Classification.
