import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Title of the app
st.title("Sentiment, Entity, and Semantic Analysis Tool")

# Add an input box for the user to type text
user_input = st.text_area("Enter the text for analysis:", height=150)

# Function for sentiment analysis
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

# Function for entity analysis using spaCy
def entity_analysis(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Function for semantic analysis (using TF-IDF for now)
def semantic_analysis(text):
    # Split text into sentences (lines) for analysis
    sentences = text.split("\n")
    
    # Vectorize the sentences using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate cosine similarity between sentences
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Prepare a DataFrame for easy display of similarities
    similarity_df = pd.DataFrame(cosine_sim, columns=[f"Line {i+1}" for i in range(len(sentences))], index=[f"Line {i+1}" for i in range(len(sentences))])
    
    return similarity_df

# Function for generating recommendations
def generate_recommendations(text):
    sentences = text.split("\n")
    recommendations = []
    
    for i, sentence in enumerate(sentences):
        # Simple recommendation based on sentence length (just for illustration)
        if len(sentence.split()) < 5:
            recommendations.append(f"Line {i+1}: Consider adding more details to make this sentence longer.")
        elif len(sentence.split()) > 20:
            recommendations.append(f"Line {i+1}: Consider shortening this sentence for better readability.")
        else:
            recommendations.append(f"Line {i+1}: Sentence length is optimal.")
    
    return recommendations

# Button to trigger analysis
if st.button("Analyze"):
    if user_input:
        # Sentiment analysis
        sentiment, sentiment_score = sentiment_analysis(user_input)
        st.write(f"Sentiment: {sentiment} (Score: {sentiment_score})")
        
        # Entity Analysis
        entities = entity_analysis(user_input)
        if entities:
            st.write("Entities Found:")
            for entity in entities:
                st.write(f"- {entity[0]} ({entity[1]})")
        else:
            st.write("No entities found.")
        
        # Semantic Analysis
        semantic_sim = semantic_analysis(user_input)
        st.write("Semantic Similarity between lines:")
        st.dataframe(semantic_sim)
        
        # Recommendations for improvement
        recommendations = generate_recommendations(user_input)
        st.write("Content Improvement Recommendations:")
        for rec in recommendations:
            st.write(rec)

        # Word Cloud generation
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=150, colormap='coolwarm').generate(user_input)

        # Display Word Cloud
        st.subheader("Word Cloud:")
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    else:
        st.warning("Please enter some text to analyze.")
