import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import re

# Use st.cache_resource to load heavy objects like models and tokenizers only once
@st.cache_resource
def load_ml_artifacts():
    # Load tokenizer
    with open('project/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load model
    model = load_model("project/emoji_model.keras", compile=False)

    # Load mapping dataframe
    mapping_df = pd.read_csv('project/Mapping.csv', encoding='utf-8')
    mapping_df.columns = mapping_df.columns.str.strip()

    # Load max_length
    with open('project/max_length.txt', 'r') as f:
        max_length_loaded = int(f.read().strip())

    return tokenizer, model, mapping_df, max_length_loaded

# Load artifacts
tokenizer, model, mapping_df, max_length = load_ml_artifacts()

# Clean text like during training
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict emoji from text
def predict_emoji(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    prediction = model.predict(padded, verbose=0)
    pred_label = int(np.argmax(prediction, axis=1)[0])

    match = mapping_df[mapping_df['Label'] == pred_label]
    if not match.empty and 'emoji' in match.columns:
        emoji = match['emoji'].values[0]
    else:
        emoji = "❓"
    return emoji

# Streamlit UI
st.set_page_config(page_title="Emoji Predictor 😊", page_icon="🤖")
st.title("🔮 Emoji Prediction App")
st.write("Enter a sentence and get a predicted emoji!")

user_input = st.text_input("Type your sentence here:")

if st.button("Predict Emoji"):
    if user_input.strip() == "":
        st.warning("Please enter something first.")
    else:
        with st.spinner("Predicting..."):
            emoji = predict_emoji(user_input)
        st.success(f"Predicted Emoji: {emoji}")
