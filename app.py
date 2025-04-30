import spacy
import streamlit as st
from collections import defaultdict

# Attempt to load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model 'en_core_web_sm' could not be found. Please install it by running the following command in your terminal:\n"
             "`python -m spacy download en_core_web_sm`")
    st.stop()  # Stop the app if the model cannot be loaded

# Recommendations dictionary
recommendations = {
    "ORG": "Consider connecting with their official website or checking their LinkedIn page.",
    "PERSON": "You might want to consider reaching out on social media platforms or sending an email.",
    "GPE": "Look for local events or services in your area related to this entity.",
    "PRODUCT": "Check out online reviews or the manufacturer's website for more details.",
}

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def generate_recommendations(entities):
    recs = defaultdict(list)
    for entity, label in entities:
        if label in recommendations:
            recs[entity].append(recommendations[label])
    return recs

def main():
    st.title("Entity Recognition Tool")
    
    input_text = st.text_area("Enter Text Here:", height=200)

    if st.button("Extract Entities"):
        entities = extract_entities(input_text)
        st.subheader("Extracted Entities:")
        for entity, label in entities:
            st.write(f" - **{entity}**: {label}")

        recs = generate_recommendations(entities)
        
        st.subheader("Recommendations:")
        for entity, rec_list in recs.items():
            for rec in rec_list:
                st.write(f" - For **'{entity}'**: {rec}")

if __name__ == "__main__":
    main()
