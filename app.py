import streamlit as st
from textblob import TextBlob
import spacy
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Load spaCy model
import en_core_web_sm
nlp = en_core_web_sm.load()

# Function for sentiment analysis and subjectivity
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"
    subjectivity = "Subjective" if blob.sentiment.subjectivity > 0.5 else "Objective"
    return sentiment, subjectivity

# Function for entity analysis
def analyze_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to generate word cloud
def generate_wordcloud(text):
    stopwords = set(["the", "and", "a", "to", "in", "of", "for", "on", "with", "as", "is", "at", "by"])
    wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, max_words=100).generate(text)
    return wordcloud

# Function to categorize sentiment and display pie chart
def sentiment_pie_chart(sentiment_data):
    sentiment_counts = Counter(sentiment_data)
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())

    # Create Pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#FF6347", "#FFD700"])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

# Streamlit UI
st.title("Sentiment and Entity Analysis")

# Input box for user text
text_input = st.text_area("Enter your text here for analysis:")

# Button to process the text
if st.button("Analyze Text"):
    if text_input:
        # Sentiment Analysis
        sentiment, subjectivity = analyze_sentiment(text_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Subjectivity: {subjectivity}")

        # Entity Analysis
        entities = analyze_entities(text_input)
        if entities:
            st.write("Entities Found:")
            for entity in entities:
                st.write(f"- {entity[0]} (Label: {entity[1]})")
        else:
            st.write("No entities found.")

        # Sentiment Pie Chart
        sentiment_pie_chart([sentiment])

        # Word Cloud
        wordcloud = generate_wordcloud(text_input)
        st.image(wordcloud.to_array(), caption="Generated Word Cloud", use_column_width=True)
    else:
        st.write("Please enter some text to analyze.")
