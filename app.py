import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import spacy
from spacy import displacy
from spacy.cli import download

# Ensure spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Streamlit layout
st.title("Text Analysis: Sentiment and Entity Analysis")
st.write("""
This app performs sentiment analysis and entity extraction from the text you provide.
It also visualizes sentiment trends and displays a word cloud.
""")

# Input box for the user to enter text
input_text = st.text_area("Enter Text for Analysis", height=200)

# Button to trigger analysis
if st.button("Analyze"):
    if input_text.strip() != "":
        # Sentiment Analysis
        blob = TextBlob(input_text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            sentiment_label = "Positive"
        elif sentiment < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        st.write(f"Sentiment Score: {sentiment}")
        st.write(f"Sentiment Label: {sentiment_label}")

        # Displaying Pie Chart for Sentiment Categorization
        sentiment_data = {"Positive": 0, "Negative": 0, "Neutral": 0}
        sentiment_data[sentiment_label] += 1
        sentiment_df = pd.DataFrame(list(sentiment_data.items()), columns=["Sentiment", "Count"])
        
        fig = px.pie(sentiment_df, names="Sentiment", values="Count", title="Sentiment Distribution")
        st.plotly_chart(fig)

        # Entity Extraction using spaCy
        doc = nlp(input_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        if entities:
            st.subheader("Extracted Entities")
            entity_df = pd.DataFrame(entities, columns=["Entity", "Label"])
            st.write(entity_df)
        else:
            st.write("No entities found in the text.")

        # Word Cloud Generation
        wordcloud = WordCloud(width=800, height=400, max_words=100, background_color="white").generate(input_text)
        
        st.subheader("Word Cloud")
        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
        
        # Recommendations (Basic Example based on Sentiment)
        st.subheader("Recommendations")
        if sentiment > 0:
            st.write("Recommendation: The text has a positive tone. Continue on the same path.")
        elif sentiment < 0:
            st.write("Recommendation: The text has a negative tone. Consider rewording for a more positive message.")
        else:
            st.write("Recommendation: The text is neutral. Try adding more engaging language.")

    else:
        st.warning("Please enter some text to analyze.")

