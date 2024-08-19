import os, validators, streamlit as st
from llama_index.core import SummaryIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader

Settings.llm = OpenAI(temperature=0.2, model="gpt-4o-mini")

# Streamlit app config
st.subheader("Summarize URL")
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", type="password")
col1, col2 = st.columns([4,1])
url = col1.text_input("URL", label_visibility="collapsed")
summarize = col2.button("Summarize")

if summarize:
    # Validate inputs
    if not openai_api_key.strip() or not url.strip():
        st.error("Please provide the missing fields.")
    elif not validators.url(url):
        st.error("Please provide a valid URL.")
    else:
        try:
            with st.spinner("Please wait..."):
                os.environ["OPENAI_API_KEY"] = openai_api_key
                documents = SimpleWebPageReader(html_to_text=True).load_data([url])
                index = SummaryIndex.from_documents(documents)

                query_engine = index.as_query_engine()
                summary = query_engine.query("Summarize the article in 200-250 words.")

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
