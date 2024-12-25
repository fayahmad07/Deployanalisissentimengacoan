import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the model and vectorizer with caching to improve loading times
@st.cache(allow_output_mutation=True)
def load_model_and_vectorizer():
    model = load('sentiment_model.joblib')
    tfidf_vectorizer = load('tfidf_vectorizer.joblib')
    return model, tfidf_vectorizer

model, tfidf_vectorizer = load_model_and_vectorizer()

def predict_sentiment(text):
    text_vector = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

def visualize_wordcloud(data):
    wc = WordCloud(width=800, height=400, background_color='white')
    fig, ax = plt.subplots()
    wordcloud = wc.generate(' '.join(data))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

st.title('Aplikasi Analisis Sentimen')

uploaded_file = st.file_uploader("Unggah file CSV yang berisi teks untuk analisis", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'comment_rests' in data.columns:
        st.write(data.head())
        if st.button('Generate WordCloud'):
            visualize_wordcloud(data['comment_rests'])
    else:
        st.error("File CSV tidak mengandung kolom 'comment_rests'. Harap unggah file dengan kolom yang benar.")

user_text = st.text_area("Masukkan teks untuk memprediksi sentimennya:", "")
if user_text:
    result = predict_sentiment(user_text)
    st.write(f"Prediksi Sentimen: {result}")  # Adjust the output to match your model's classes

if uploaded_file is not None and 'sentimen_multi' in data.columns and st.button('Tampilkan Distribusi Sentimen'):
    sentiment_distribution = data['sentimen_multi'].value_counts(normalize=True)
    fig, ax = plt.subplots()
    ax.pie(sentiment_distribution, labels=sentiment_distribution.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
else:
    st.error("File CSV tidak mengandung kolom 'sentimen_multi'.")
