import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Title of the app
st.title("Sentiment Analysis and Word Cloud Generator")

# Add an input box for the user to type text
user_input = st.text_area("Enter the text for analysis:", height=150)

# Button to trigger analysis
if st.button("Analyze"):
    if user_input:
        # Sentiment Analysis function
        def sentiment_analysis(text):
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            if sentiment_score > 0:
                sentiment = "Positive"
            elif sentiment_score < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            return sentiment, sentiment_score

        # Perform sentiment analysis on the input text
        sentiment, sentiment_score = sentiment_analysis(user_input)

        # Display Sentiment Result
        st.write(f"Sentiment: {sentiment} (Score: {sentiment_score})")

        # Word Cloud generation
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=150, colormap='coolwarm').generate(user_input)

        # Display Word Cloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    else:
        st.warning("Please enter some text to analyze.")
