import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import spacy
import subprocess
import sys
import importlib

# Ensure spaCy model is downloaded
try:
    importlib.import_module("en_core_web_sm")
except ImportError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Semantic & Entity Analyzer", layout="wide")
st.title("🧠 Semantic Analyzer with Sentiment, Entities & Recommendations")

# User input with submit button
with st.form("text_form"):
    text = st.text_area("Enter text to analyze", height=300)
    submitted = st.form_submit_button("Analyze")

if submitted and text.strip():
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Determine sentiment category
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Show sentiment
    st.subheader("Sentiment Summary")
    st.metric("Polarity", f"{polarity:.2f}")
    st.metric("Subjectivity", f"{subjectivity:.2f}")
    st.metric("Category", sentiment)

    # Pie chart
    st.subheader("Sentiment Category Pie Chart")
    fig1 = px.pie(
        names=["Positive", "Neutral", "Negative"],
        values=[
            1 if sentiment == "Positive" else 0,
            1 if sentiment == "Neutral" else 0,
            1 if sentiment == "Negative" else 0,
        ],
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # WordCloud
    st.subheader("Word Cloud")
    stopwords = set(STOPWORDS)
    wc = WordCloud(
        background_color="white",
        stopwords=stopwords,
        max_words=200,
        width=800,
        height=400
    ).generate(text)

    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig2)

    # Entity analysis
    st.subheader("Named Entity Recognition")
    doc = nlp(text)
    if doc.ents:
        entity_data = [(ent.text, ent.label_) for ent in doc.ents]
        df_entities = pd.DataFrame(entity_data, columns=["Entity", "Label"])
        st.dataframe(df_entities)
    else:
        st.info("No entities found in the text.")

    # Basic Recommendations
    st.subheader("Recommendations")
    recommendations = []
    if subjectivity > 0.5:
        recommendations.append("Try to make your content more objective.")
    else:
        recommendations.append("Content has good objectivity.")
    
    if polarity < -0.3:
        recommendations.append("Consider revising negative tone for a more balanced message.")
    elif polarity > 0.3:
        recommendations.append("Positive tone detected — great for persuasive or promotional content!")

    if not recommendations:
        recommendations.append("Content sentiment is balanced and neutral.")

    for rec in recommendations:
        st.write(f"👉 {rec}")
