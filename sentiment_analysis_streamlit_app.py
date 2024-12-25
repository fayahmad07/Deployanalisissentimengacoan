
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Dictionary for text normalization
replacements = {
    'gmna': 'bagaimana', 'yng': 'yang', 'jgn': 'jangan', 'tdk': 'tidak', 'aja': 'saja',
    'udh': 'sudah', 'lg': 'lagi', 'dg': 'dengan', 'tp': 'tapi', 'sm': 'sama', 'klo': 'kalau',
    'krn': 'karena', 'bgt': 'banget', 'utk': 'untuk', 'blm': 'belum', 'gk': 'tidak',
    'dgn': 'dengan', 'ny': 'nya', 'gw': 'saya', 'yg': 'yang', 'n': 'dan', 'lu': 'kamu',
    'gue': 'saya', 'lo': 'kamu', 'trus': 'terus', 'kl': 'kalau', 'd': 'di', 'msh': 'masih',
    'bnyk': 'banyak', 'jg': 'juga', 'dlu': 'dulu', 'dll': 'dan lain-lain', 'bs': 'bisa'
}

# Function for text normalization
def normalize_text(data):
    data['text_column'] = data['text_column'].replace(replacements, regex=True)
    return data

# Function to impute missing values with RandomForest
def impute_missing_values(data, target_column, predictor_columns):
    train_data = data[data[target_column].notna()]
    predict_data = data[data[target_column].isna()]

    rf = RandomForestRegressor(random_state=42)
    rf.fit(train_data[predictor_columns], train_data[target_column])
    predicted_values = rf.predict(predict_data[predictor_columns])

    data.loc[data[target_column].isna(), target_column] = predicted_values
    return data

# Sentiment analysis function (placeholder)
def analyze_sentiment(data):
    return np.random.choice(['Positive', 'Negative', 'Neutral'], size=len(data))

# Main function for Streamlit app
def main():
    st.title("Sentiment Analysis with CSV Upload")
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Preprocessing
        data = normalize_text(data)
        data = impute_missing_values(data, 'your_target_column', ['your_predictor_columns'])
        
        # Sentiment Analysis
        data['Sentiment'] = analyze_sentiment(data)

        # Pie Chart Visualization
        fig, ax = plt.subplots()
        sentiment_counts = data['Sentiment'].value_counts()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

        # Word Cloud
        text = " ".join(word for word in data['text_column'])
        wordcloud = WordCloud(background_color='white').generate(text)
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()

if __name__ == "__main__":
    main()
