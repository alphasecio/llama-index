import validators, streamlit as st
from llama_index.core import SummaryIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.web import SimpleWebPageReader

# Streamlit app config
st.set_page_config(page_title="Summarize URL")
st.subheader("Summarize URL")
st.markdown("Enter a URL to generate a concise summary of the content.")

# Initialize session state to store Google API key
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

with st.sidebar:
    st.header("Settings")
    google_api_key = st.text_input("Google API key", value=st.session_state.google_api_key, type="password")

    if google_api_key and google_api_key != st.session_state.google_api_key:
        st.session_state.google_api_key = google_api_key

col1, col2 = st.columns([4,1])
url = col1.text_input("URL", placeholder="https://example.com", label_visibility="collapsed")
summarize = col2.button("Summarize")

if summarize:
    # Validate inputs
    if not google_api_key.strip():
        st.error("Please provide the Google API key.")
    if not url.strip():
        st.error("Please provide the URL to summarize.")
    elif not validators.url(url):
        st.error("Please provide a valid URL (including https://).")
    else:
        try:
            with st.spinner("Fetching content and generating summary...", show_time=True):
                Settings.llm = GoogleGenAI(model="gemini-2.0-flash", api_key=google_api_key)
                documents = SimpleWebPageReader(html_to_text=True).load_data([url])

                if not documents:
                    st.error("Failed to retrieve content from the URL.")
                else:
                    index = SummaryIndex.from_documents(documents)
                    query_engine = index.as_query_engine()
                    summary = query_engine.query("Summarize the article in 200-250 words.")
                    st.success(summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            with st.expander("Error Details"):
                st.exception(e)
