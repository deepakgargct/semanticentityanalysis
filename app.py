import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob

st.set_page_config(page_title="Dandelion NLP App", layout="centered")

st.title("🧠 NLP with Dandelion + TextBlob")
st.write("This app performs **Entity Extraction** using Dandelion.eu and **Sentiment Analysis** using TextBlob.")

token = st.text_input("🔐 Enter your Dandelion API Token", type="password")

if token:
    datatxt = DataTXT(token=token)
    user_input = st.text_area("✏️ Enter text to analyze:", height=200)

    tasks = st.multiselect("🛠️ Select NLP tasks", ["Entity Extraction", "Sentiment Analysis"])

    if st.button("🔍 Analyze") and user_input:
        with st.spinner("Analyzing..."):
            lang_result = datatxt.li(user_input)
            lang = lang_result.get("detectedLangs", [{}])[0].get("lang", "unknown")
            st.success(f"Detected Language: `{lang}`")

            if "Entity Extraction" in tasks:
                st.subheader("📌 Named Entity Extraction")
                try:
                    nex_result = datatxt.nex(user_input, include="types,abstract,categories")
                    if nex_result.annotations:
                        for ann in nex_result.annotations:
                            st.markdown(f"- **{ann.label}** (URI: {ann.uri}) — {ann.categories or 'No category'}")
                    else:
                        st.info("No entities found.")
                except Exception as e:
                    st.error(f"Entity extraction error: {e}")

            if "Sentiment Analysis" in tasks:
                st.subheader("❤️ Sentiment Analysis (TextBlob)")
                try:
                    blob = TextBlob(user_input)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity

                    st.write(f"**Polarity:** {polarity:.2f} (range: -1 to 1)")
                    st.write(f"**Subjectivity:** {subjectivity:.2f} (range: 0 to 1)")

                    sentiment_type = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
                    st.success(f"Overall Sentiment: **{sentiment_type}**")
                except Exception as e:
                    st.error(f"Sentiment analysis error: {e}")
else:
    st.warning("Please enter your Dandelion API token to begin.")
