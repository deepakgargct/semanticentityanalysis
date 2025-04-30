import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd
import numpy as np

# Sentiment Analysis function using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return sentiment, subjectivity

# Entity Extraction using TextBlob
def extract_entities(text):
    blob = TextBlob(text)
    return [(word, 'NOUN') for word, pos in blob.tags if pos == 'NN' or pos == 'NNS']

# Visualization function for Word Cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, max_words=200).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot()

# Visualization function for Sentiment Trend (Pie Chart)
def plot_sentiment_pie(sentiments):
    positive = sum(1 for sentiment in sentiments if sentiment > 0)
    negative = sum(1 for sentiment in sentiments if sentiment < 0)
    neutral = len(sentiments) - positive - negative

    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive, negative, neutral]
    
    fig = px.pie(values=values, names=labels, title="Sentiment Distribution")
    st.plotly_chart(fig)

# Streamlit UI
st.title('Text Sentiment and Entity Analysis')

# Input Text Box
input_text = st.text_area("Enter your text for analysis:")

# Button to trigger processing
if st.button('Analyze'):
    if input_text:
        # Analyze Sentiment
        sentiment, subjectivity = analyze_sentiment(input_text)
        st.write(f"Sentiment (Polarity): {sentiment}")
        st.write(f"Subjectivity: {subjectivity}")

        # Display sentiment trend chart (Positive, Negative, Neutral)
        sentiments = [sentiment]  # Replace with multiple sentiment scores for trends
        plot_sentiment_pie(sentiments)

        # Generate Word Cloud
        generate_word_cloud(input_text)

        # Extract and display Entities
        entities = extract_entities(input_text)
        st.write(f"Extracted Entities: {entities}")
    else:
        st.write("Please enter some text for analysis.")
