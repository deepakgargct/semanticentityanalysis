import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

st.title("Sentiment Analysis and Word Cloud")

user_input = st.text_area("Enter Text for Analysis")

if user_input:
    sentiment = sentiment_analysis(user_input)
    st.write(f"Sentiment: {sentiment}")

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
