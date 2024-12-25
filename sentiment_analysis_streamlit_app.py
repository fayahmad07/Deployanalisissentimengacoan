import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load model dan vectorizer
model = load('sentiment_model.joblib')
tfidf_vectorizer = load('tfidf_vectorizer.joblib')

def predict_sentiment(text):
    """Fungsi untuk memprediksi sentimen dari teks."""
    text_vector = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

def visualize_wordcloud(data):
    """Fungsi untuk visualisasi WordCloud."""
    wc = WordCloud(width=800, height=400, background_color='white')
    fig, ax = plt.subplots()
    wordcloud = wc.generate(' '.join(data))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def visualize_pie_chart(sentiments):
    """Fungsi untuk visualisasi pie chart distribusi sentimen."""
    fig, ax = plt.subplots()
    sentiment_counts = sentiments.value_counts(normalize=True)
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

st.title('Aplikasi Analisis Sentimen')

# Form untuk input teks
with st.form("text_input"):
    user_text = st.text_area("Masukkan teks untuk memprediksi sentimennya:")
    submit_text = st.form_submit_button("Prediksi Sentimen")

if submit_text and user_text:
    result = predict_sentiment(user_text)
    st.write(f"Prediksi Sentimen: {result}")

# Pengunggahan file CSV dan visualisasi
uploaded_file = st.file_uploader("Unggah file CSV yang berisi teks untuk analisis", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'comment_rests' in data.columns and 'sentimen_multi' in data.columns:
        st.write(data.head())
        if st.button('Tampilkan WordCloud dari Komentar'):
            visualize_wordcloud(data['comment_rests'])
        if st.button('Tampilkan Distribusi Sentimen'):
            visualize_pie_chart(data['sentimen_multi'])
    else:
        st.error("Pastikan file CSV memiliki kolom 'comment_rests' dan 'sentimen_multi'.")
