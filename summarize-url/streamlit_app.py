import ipaddress
import logging
import os
import socket
from urllib.parse import urlsplit

import requests
import streamlit as st
from google.genai import errors as genai_errors
from llama_index.core import Document, SummaryIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.web import SimpleWebPageReader

logger = logging.getLogger(__name__)


def env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, default))
        return value if value > 0 else default
    except ValueError:
        return default


# Defaults can be overridden with environment variables
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash").strip()
MAX_CHARS = env_int("MAX_CHARS", 60000)
FETCH_TIMEOUT = 20
SUMMARY_PROMPT = "Summarize the article in 200-250 words."


class NoContentError(Exception):
    """Raised when the page has no readable text."""


def check_public_url(url: str) -> str | None:
    """Return an error message if the URL is not a public http(s) address, else None."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "Please provide a valid URL (including https://)."
    if parts.username or parts.password:
        return "URLs with embedded credentials are not allowed."
    try:
        port = parts.port
    except ValueError:
        return "Please provide a valid URL."
    if port not in (None, 80, 443):
        return "Only standard ports (80 and 443) are allowed."
    try:
        infos = socket.getaddrinfo(parts.hostname, port or 443, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        return "Could not resolve the host name."
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.version == 6:
            if ip.sixtofour or ip.teredo:
                return "Tunnelled IPv6 addresses are not allowed."
            if ip.ipv4_mapped:
                ip = ip.ipv4_mapped
        if not ip.is_global:
            return "URLs pointing to private or internal addresses are not allowed."
    return None


def friendly_error(e: Exception) -> str:
    if isinstance(e, NoContentError):
        return "No readable content was found at this URL."
    if isinstance(e, genai_errors.APIError):
        message = (e.message or "").lower()
        if e.code == 400 and "api key" in message:
            return "Google rejected the API key."
        if e.code in (401, 403):
            return "The API key has no access to this model."
        if e.code == 404:
            return "The Gemini model was not found."
        if e.code == 429:
            return "Gemini rate limit or quota exceeded."
        return f"Gemini returned an error (HTTP {e.code})."
    if isinstance(e, requests.RequestException):
        return "Could not fetch the page. Check the URL and try again."
    if isinstance(e, ValueError) and "Error fetching page" in str(e):
        return "The site returned an error instead of the page."
    return f"Unexpected error ({type(e).__name__})."


# Streamlit app config
st.set_page_config(page_title="Summarize URL", page_icon="🔗")

with st.sidebar:
    st.header("Settings")
    st.text_input("Google API key", type="password", key="google_api_key")
    st.caption(
        "Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey). "
        "Keys stay in your browser session only."
    )
    st.divider()
    st.text_input("Gemini model", value=DEFAULT_MODEL, key="gemini_model")

st.header("🔗 Summarize URL")
st.caption("Fetch a web page, index it with LlamaIndex and summarize it with Google Gemini.")

with st.form("url_form", border=False):
    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
    url = col1.text_input(
        "URL",
        placeholder="https://example.com",
        label_visibility="collapsed",
        max_chars=2048,
    )
    summarize = col2.form_submit_button("Summarize", type="primary", width="stretch")

if not summarize:
    st.info("Add your Google API key in the sidebar, paste a URL and click Summarize.")
    st.stop()

google_api_key = st.session_state.google_api_key.strip()
model = st.session_state.gemini_model.strip()
url = url.strip()

if not google_api_key:
    st.error("Please provide the Google API key.")
elif not model:
    st.error("Please provide a Gemini model name.")
elif not url:
    st.error("Please provide a URL.")
elif url_error := check_public_url(url):
    st.error(url_error)
else:
    try:
        with st.spinner("Fetching content and generating summary...", show_time=True):
            llm = GoogleGenAI(model=model, api_key=google_api_key)
            documents = SimpleWebPageReader(
                html_to_text=True, timeout=FETCH_TIMEOUT, fail_on_error=True
            ).load_data([url])
            text = documents[0].text.strip() if documents else ""
            if not text:
                raise NoContentError()

            truncated = len(text) > MAX_CHARS
            document = Document(text=text[:MAX_CHARS], metadata={"url": url})
            index = SummaryIndex.from_documents([document])
            query_engine = index.as_query_engine(llm=llm, response_mode="tree_summarize")
            summary = query_engine.query(SUMMARY_PROMPT)
    except Exception as e:
        logger.exception("Summarization failed")
        st.error(friendly_error(e))
        st.stop()

    with st.container(border=True):
        st.markdown("**Summary**")
        st.markdown(summary.response or "_No summary returned._")
        note = f"Source: {urlsplit(url).hostname} · {len(document.text.split()):,} words read"
        if truncated:
            note += f" · page truncated to the first {MAX_CHARS:,} characters"
        st.caption(note)
