# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Streamlit UI Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")

# --- Configuration (must match training config) ---
MAX_FEATURES = 10000
MAX_LEN = 250
# --- UPDATED MODEL PATH ---
MODEL_PATH = 'sentiment_bidirectional_lstm_model.h5'
# -------------------------
WORD_INDEX_PATH = 'word_index.json'

# --- Load NLTK data (if not already downloaded) ---

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Load Model and Word Index ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model_and_word_index():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(WORD_INDEX_PATH, 'r') as f:
        word_index = json.load(f)
    return model, word_index

model, word_index = load_model_and_word_index()

# --- Preprocessing function (must match training preprocessing) ---
def preprocess_text(text):
    text = text.lower() # Lowercase
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words] # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # Lemmatize
    return tokens

def encode_and_pad(text_tokens, word_index, max_len=MAX_LEN, max_features=MAX_FEATURES):
    encoded_review = []
    # Keras IMDb index starts from 3 (0: padding, 1: start, 2: unknown)
    # So we add 3 to the word_index value.
    for word in text_tokens:
        if word in word_index and word_index[word] + 3 < max_features:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2) # Unknown word token

    padded_review = pad_sequences([encoded_review], maxlen=max_len, padding='pre', truncating='pre')
    return padded_review[0] # Return the single padded sequence

# --- Streamlit UI ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")

user_review = st.text_area("Your Movie Review:", height=150, placeholder="Type your movie review here...")

if st.button("Analyze Sentiment"):
    if user_review:
        with st.spinner("Analyzing..."):
            # 1. Preprocess the input text
            processed_tokens = preprocess_text(user_review)
            # st.write(f"Processed tokens: {processed_tokens}") # Can uncomment for debugging

            # 2. Encode and pad the sequence
            input_sequence = encode_and_pad(processed_tokens, word_index)
            # Reshape for model prediction (batch size of 1)
            input_sequence = np.array([input_sequence])

            # 3. Make prediction
            prediction = model.predict(input_sequence)[0][0]

            # 4. Display result
            st.markdown("---")
            if prediction >= 0.5:
                sentiment = "Positive"
                st.success(f"**Predicted Sentiment: {sentiment}** (Confidence: {prediction:.2f})")
                st.balloons()
            else:
                sentiment = "Negative"
                st.error(f"**Predicted Sentiment: {sentiment}** (Confidence: {1 - prediction:.2f})")
            st.markdown("---")
            st.info("A confidence score close to 1.0 indicates strong sentiment, while close to 0.5 indicates uncertainty.")
    else:
        st.warning("Please enter a review to analyze.")

st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
This app uses a **Bidirectional LSTM Neural Network** (an improvement over Simple RNN) trained on the IMDb movie review dataset.
The process involves:
1.  **Text Preprocessing:** Cleaning the input text (lowercasing, removing punctuation/numbers, stopwords, lemmatization).
2.  **Word Embedding:** Converting words into numerical vectors that capture their meaning.
3.  **LSTM Analysis:** The Bidirectional LSTM processes the sequence of word embeddings, effectively understanding context from both directions, which helps in capturing nuanced sentiment.
4.  **Sentiment Prediction: ** The model outputs a probability (0-1) indicating positive sentiment.
""")