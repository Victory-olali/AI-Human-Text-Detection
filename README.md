# AI vs Human Text Detection

## Overview

This project develops and evaluates machine learning and deep learning models for detecting AI-generated text. The system compares multiple feature extraction techniques and classification models to determine which approaches are most effective for distinguishing between human-written and AI-generated content.

The project includes:

* TF-IDF feature extraction
* GloVe word embeddings
* Support Vector Machine (SVM)
* Decision Tree
* AdaBoost
* Feedforward Neural Network (FNN)
* Long Short-Term Memory (LSTM)
* Convolutional Neural Network (CNN)

A Streamlit web application was developed to allow users to enter text and receive predictions from the trained models.

## Dataset

The dataset consists of text samples labeled as either:

* 0 = Human Written
* 1 = AI Generated

The data was preprocessed using:

* Punctuation removal
* Stopword removal
* Tokenization
* Lemmatization

## Results Summary

TF-IDF features consistently outperformed GloVe embeddings across all evaluated models.

| Model         | TF-IDF Accuracy |
| ------------- | --------------- |
| SVM           | 97%             |
| Decision Tree | 90%             |
| AdaBoost      | 95%             |
| FNN           | 97%             |
| LSTM          | 82%             |
| CNN           | 97%             |

The CNN achieved the highest ROC-AUC score of 0.995, making it the strongest overall model.

## Running the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```text
ai_human_detection_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── svm_model.pkl
│   ├── decision_tree_model.pkl
│   ├── adaboost_model.pkl
│   ├── fnn_model.keras
│   ├── lstm_model.keras
│   ├── cnn_model.keras
│   ├── tokenizer.pkl
│   └── max_len.pkl
│
└── data/
    ├── training_data/
    |   ├── train_data with labels.csv
    |   └── train_data.csv
    └── test_data/
        └── test_data.csv
```

## Author

Olali Victory

