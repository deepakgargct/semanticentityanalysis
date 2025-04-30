import spacy
import streamlit as st
from collections import defaultdict

# Attempt to load the spaCy model, and handle the error when it doesn't exist
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None  # Set nlp to None if model fails to load

# Recommendations dictionary
recommendations = {
    "ORG": "Consider connecting with their official website or checking their LinkedIn page.",
    "PERSON": "You might want to consider reaching out on social media platforms or sending an email.",
    "GPE": "Look for local events or services in your area related to this entity.",
    "PRODUCT": "Check out online reviews or the manufacturer's website for more details.",
}

def extract_entities(text):
    if nlp is None:
        return []  # Return an empty list if spaCy model is not loaded
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
        if nlp is None:
            st.error("Unable to load the spaCy model 'en_core_web_sm'. Please install it in your local environment.")
            return
        
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
