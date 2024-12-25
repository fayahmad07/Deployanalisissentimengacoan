import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from googletrans import Translator
import emoji

# Fungsi untuk mengatasi emoji
def handle_emojis(data, column):
    data[column] = data[column].apply(lambda text: emoji.demojize(text))
    return data

# Fungsi untuk membersihkan teks
def clean_text(data, column):
    data[column] = data[column].apply(lambda text: text.lower())  # Lowercase
    data[column] = data[column].apply(lambda text: re.sub(r'[^\w\s]', '', text))  # Remove punctuation
    data[column] = data[column].apply(lambda text: re.sub(r'\d+', '', text))  # Remove numbers
    data[column] = data[column].apply(lambda text: re.sub(r'\s+', ' ', text).strip())  # Remove extra whitespaces
    return data

# Fungsi untuk menghapus karakter non-latin (termasuk Jepang, Korea, Kanji)
def remove_non_latin_characters(data, column):
    data[column] = data[column].apply(lambda text: re.sub(r'[^\x00-\x7F]+', ' ', text))
    return data

# Fungsi untuk translate teks ke Bahasa Inggris
def translate_text(data, column):
    translator = Translator()
    data[column] = data[column].apply(lambda text: translator.translate(text, dest='en').text)
    return data

# Fungsi untuk mengatasi kata yang tidak bermakna, stop words, dan stemming
def preprocess_text(data, column):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    data[column] = data[column].apply(lambda text: ' '.join(stemmer.stem(word) for word in text.split() if word not in stop_words))
    return data

# Fungsi untuk vektorisasi teks
def vectorize_text(data, column):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(data[column])
    feature_names = vectorizer.get_feature_names_out()
    return pd.DataFrame(vectors.toarray(), columns=feature_names)

# Main function untuk aplikasi Streamlit
def main():
    st.title("Comprehensive Text Preprocessing")
    uploaded_file = st.file_uploader("Upload your CSV file", type='csv')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        text_column = st.selectbox("Select column to process", options=data.columns)
        
        # Preprocessing steps
        data = handle_emojis(data, text_column)
        data = clean_text(data, text_column)
        data = remove_non_latin_characters(data, text_column)
        data = translate_text(data, text_column)
        data = preprocess_text(data, text_column)
        vectors = vectorize_text(data, text_column)
        
        st.write("Processed Data:", vectors.head())
        # Option to download the processed data as CSV
        st.download_button("Download Processed CSV", data.to_csv().encode('utf-8'), "text/csv", "processed_data.csv")

if __name__ == "__main__":
    main()
