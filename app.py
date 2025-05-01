import streamlit as st
from dandelion import DataTXT
import os

# Streamlit page config
st.set_page_config(page_title="Dandelion NLP App", layout="centered")

st.title("🧠 NLP with Dandelion.eu API")
st.write("This app performs **Entity Extraction** and **Sentiment Analysis** using [Dandelion.eu](https://dandelion.eu).")

# Input API Token
token = st.text_input("🔐 Enter your Dandelion API Token", type="password")

# Validate token
if token:
    datatxt = DataTXT(token=token)

    # Text input
    user_input = st.text_area("✏️ Enter text to analyze:", height=200)

    if user_input:
        # Select tasks
        tasks = st.multiselect("🛠️ Select NLP tasks", ["Entity Extraction", "Sentiment Analysis"])

        if st.button("🔍 Analyze"):
            with st.spinner("Analyzing..."):

                # Language detection
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
                    st.subheader("❤️ Sentiment Analysis")
                    try:
                        sentiment = datatxt.sent(user_input)
                        sentiment_score = sentiment.get("sentiment", {}).get("score", 0.0)
                        sentiment_type = sentiment.get("sentiment", {}).get("type", "unknown")
                        st.write(f"**Sentiment:** {sentiment_type.capitalize()} ({sentiment_score:.2f})")
                    except Exception as e:
                        st.error(f"Sentiment analysis error: {e}")
else:
    st.warning("Please enter your Dandelion API token to begin.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Dandelion.eu")
