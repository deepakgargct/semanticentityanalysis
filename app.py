import spacy
import streamlit as st
from collections import defaultdict

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Recommendations dictionary
recommendations = {
    "ORG": "Consider connecting with their official website or checking their LinkedIn page.",
    "PERSON": "You might want to consider reaching out on social media platforms or sending an email.",
    "GPE": "Look for local events or services in your area related to this entity.",
    "PRODUCT": "Check out online reviews or the manufacturer's website for more details.",
}

def extract_entities(text):
    """
    Extract named entities from the text using spaCy's NER.
    
    Args:
        text (str): The input text from which entities are to be extracted.

    Returns:
        list: A list of tuples containing (entity text, entity label).
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def generate_recommendations(entities):
    """
    Generate recommendations based on the extracted entities.

    Args:
        entities (list): A list of tuples containing (entity text, entity label).

    Returns:
        defaultdict: A dictionary of recommendations.
    """
    recs = defaultdict(list)

    for entity, label in entities:
        if label in recommendations:
            recs[entity].append(recommendations[label])

    return recs

def main():
    st.title("Entity Recognition Tool")
    
    # User input
    input_text = st.text_area("Enter Text Here:", height=200)

    if st.button("Extract Entities"):
        # Step 1: Extract entities
        entities = extract_entities(input_text)
        st.subheader("Extracted Entities:")
        for entity, label in entities:
            st.write(f" - **{entity}**: {label}")

        # Step 2: Generate Recommendations
        recs = generate_recommendations(entities)
        
        st.subheader("Recommendations:")
        for entity, rec_list in recs.items():
            for rec in rec_list:
                st.write(f" - For **'{entity}'**: {rec}")

if __name__ == "__main__":
    main()
