import streamlit as st
from textblob import TextBlob
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

st.title("Text Analysis App")
st.markdown("This app performs **Sentiment Analysis**, **Entity Recognition**, and gives **Recommendations**.")

# Text input and button
user_input = st.text_area("Enter your text for analysis:", height=200)
if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Sentiment Analysis
        blob = TextBlob(user_input)
        sentiment_polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if sentiment_polarity > 0:
            sentiment_label = "Positive"
        elif sentiment_polarity < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        st.subheader("Sentiment Analysis")
        st.write(f"**Polarity:** {sentiment_polarity:.2f}")
        st.write(f"**Subjectivity:** {subjectivity:.2f}")
        st.write(f"**Sentiment Label:** {sentiment_label}")

        # Pie Chart
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [0, 0, 0]
        if sentiment_label == "Positive": sizes[0] = 1
        elif sentiment_label == "Negative": sizes[1] = 1
        else: sizes[2] = 1

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'red', 'gray'])
        ax1.axis('equal')
        st.pyplot(fig1)

        # Word Cloud
        st.subheader("Word Cloud")
        wordcloud = WordCloud(width=800, height=400, stopwords=STOPWORDS, background_color='white').generate(user_input)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)

        # Entity Analysis
        st.subheader("Entity Recognition")
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            entity_df = pd.DataFrame(entities, columns=["Entity", "Label"])
            st.dataframe(entity_df)
        else:
            st.info("No named entities found.")

        # Recommendations
        st.subheader("Recommendations")
        recommendations = []
        if sentiment_label == "Positive":
            recommendations.append("✅ Consider amplifying this content on social channels.")
        elif sentiment_label == "Negative":
            recommendations.append("⚠️ Review for possible issues or complaints.")

        for ent, label in entities:
            if label in ["ORG", "PERSON"]:
                recommendations.append(f"📌 Mentioned entity '{ent}' may be of strategic importance.")

        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.write("No specific recommendations found.")
