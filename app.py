import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st
import pickle
import string
import nltk
import pandas as pd
from transformers import pipeline
import docx
import PyPDF2

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources if needed
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("wordnet")

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_keras(path):
    return load_model(path)

@st.cache_resource
def load_distilbert():
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_roberta():
    return pipeline(
        "text-classification",
        model="openai-community/roberta-base-openai-detector"
    )

def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def read_docx(file):
    document = docx.Document(file)
    return "\n".join([para.text for para in document.paragraphs])

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

# ==========================
# PREPROCESSING
# ==========================

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if not isinstance(text, str):
        return ""

    STOPWORDS = stopwords.words("english") + [
        "u", "ü", "ur", "4", "2",
        "im", "dont", "doin", "ure", "id"
    ]

    # Remove punctuation
    nopunc = "".join(
        [char for char in text if char not in string.punctuation]
    )

    # Tokenize
    tokens = word_tokenize(nopunc)

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word.lower() not in STOPWORDS
    ]

    # Lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ==========================
# LOAD MODELS
# ==========================

vectorizer = load_pickle("models/tfidfvectorizer.pkl")
tokenizer = load_pickle("models/tokenizer.pkl")

results = pd.DataFrame({
    "Model": ["SVM", "Decision Tree", "AdaBoost", "FNN", "LSTM", "CNN","DistilBERT", "RoBERTa"],
    "Accuracy": [0.97, 0.90, 0.95, 0.97, 0.81, 0.96, None, None],
    "Precision": [0.97, 0.91, 0.96, 0.97, 0.83, 0.97, None, None],
    "Recall": [0.97, 0.89, 0.95, 0.97, 0.80, 0.97, None, None],
    "F1 Score": [0.97, 0.90, 0.95, 0.97, 0.82, 0.97, None, None],
    "ROC AUC": [0.971, 0.902, 0.95, 0.97, 0.87, 0.995, None, None]
})


# ==========================
# UI
# ==========================

st.title("AI vs Human Text Classifier")

model_choice = st.selectbox(
    "Choose Model",
    ["SVM", "Decision Tree", "AdaBoost", "FNN", "LSTM", "CNN","DistilBERT", "RoBERTa"]
)

uploaded_file = st.file_uploader(
    "Upload a file",
    type=["txt", "docx", "pdf"]
)

text = ""

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "txt":
        text = read_txt(uploaded_file)

    elif file_type == "docx":
        text = read_docx(uploaded_file)

    elif file_type == "pdf":
        text = read_pdf(uploaded_file)

    st.text_area("Extracted text", text, height=250)

else:
    text = st.text_area("Enter text", height=200)


# ==========================
# PREDICTION
# ==========================

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    processed_text = preprocess(text)

    probability = None
    confidence = None

    # TF-IDF based models
    if model_choice in ["SVM", "Decision Tree", "AdaBoost", "FNN"]:

        X = vectorizer.transform([processed_text])

        if model_choice == "SVM":
            svm = load_pickle("models/svm.pkl")
            prediction = svm.predict(X)[0]

            if hasattr(svm, "predict_proba"):
                probability = svm.predict_proba(X)[0][1]

        elif model_choice == "Decision Tree":
            decision_tree = load_pickle("models/decision_tree.pkl")
            prediction = decision_tree.predict(X)[0]
            probability = decision_tree.predict_proba(X)[0][1]

        elif model_choice == "AdaBoost":
            adaboost = load_pickle("models/adaboost.pkl")
            prediction = adaboost.predict(X)[0]
            probability = adaboost.predict_proba(X)[0][1]

        elif model_choice == "FNN":
            fnn_model = load_keras("models/fnn_model.keras")
            fnn_probs = fnn_model.predict(X.toarray())[0]
            prediction = fnn_probs.argmax()
            probability = fnn_probs[1]


    # LSTM / CNN models
    elif model_choice in ["LSTM", "CNN"]:

        sequence = tokenizer.texts_to_sequences([processed_text])

        padded = pad_sequences(
            sequence,
            maxlen=200,
            padding="post",
            truncating="post"
        )

        if model_choice == "LSTM":
            lstm_model = load_keras("models/lstm_model.keras")
            lstm_probs = lstm_model.predict(padded)[0]

            prediction = lstm_probs.argmax()
            probability = lstm_probs[1]

        elif model_choice == "CNN":
            cnn_model = load_keras("models/cnn_model.keras")
            probability = cnn_model.predict(padded)[0][0]
            prediction = 1 if probability >= 0.5 else 0

    # Hugging Face models
    elif model_choice == "DistilBERT":

        distilbert = load_distilbert()
        result = distilbert(text)[0]

        hf_label = result["label"]
        confidence = result["score"]

        # Your current DistilBERT is sentiment-based:
        # POSITIVE/NEGATIVE is NOT truly AI/Human.
        # This mapping is just a placeholder.
        if hf_label.lower() in ["ai", "fake", "generated", "label_1", "positive"]:
            prediction = 1
        else:
            prediction = 0

    elif model_choice == "RoBERTa":

        roberta = load_roberta()
        result = roberta(text)[0]

        hf_label = result["label"]
        confidence = result["score"]

        if hf_label.lower() == "fake":
            prediction = 1
        else:
            prediction = 0

    # Display result
    label = "AI Generated" if prediction == 1 else "Human Written"

    st.subheader("Prediction")
    st.success(label)

    if confidence is None and probability is not None:
        if prediction == 1:
            confidence = probability
        else:
            confidence = 1 - probability

    if confidence is not None:
        st.write(f"Confidence: {confidence * 100:.2f}%")

    st.write("Processed text:")
    st.code(processed_text)


    # ==========================
    # SUMMARY REPORT
    # ==========================
    
    st.subheader("Model Performance Summary")
    
    st.dataframe(results)
    
    best_model = results.loc[results["Accuracy"].idxmax(), "Model"]
    best_accuracy = results["Accuracy"].max()
    
    st.success(f"Best Model: {best_model} ({best_accuracy:.2%} accuracy)")
    
    csv = results.to_csv(index=False)

    st.download_button(
    label="Download Summary Report",
    data=csv,
    file_name="model_summary_report.csv",
    mime="text/csv"
    )
