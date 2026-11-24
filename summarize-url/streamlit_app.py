import ipaddress
import logging
import os
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

import html2text
import httpx
import streamlit as st
from google.genai import errors as genai_errors
from llama_index.core import Document, SummaryIndex
from llama_index.llms.google_genai import GoogleGenAI

logger = logging.getLogger(__name__)


def env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, default))
        return value if value > 0 else default
    except ValueError:
        return default


# Defaults can be overridden with environment variables
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
MAX_CHARS = env_int("MAX_CHARS", 60000)
FETCH_TIMEOUT = 20
MAX_REDIRECTS = 5
MAX_BYTES = 5_000_000
USER_AGENT = "Mozilla/5.0 (compatible; summarize-url/1.0)"
SUMMARY_PROMPT = "Summarize the article in 200-250 words."


class NoContentError(Exception):
    """Raised when the page has no readable text."""


class UrlError(Exception):
    """Raised when a URL is not a public http(s) address; the message is shown to the user."""


class FetchError(Exception):
    """Raised when the page could not be fetched; the message is shown to the user."""


def resolve_public(url: str) -> tuple[str, str, int]:
    """Validate a URL and return (hostname, a vetted public IP, port), or raise UrlError."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UrlError("Please provide a valid URL (including https://).")
    if parts.username or parts.password:
        raise UrlError("URLs with embedded credentials are not allowed.")
    try:
        port = parts.port
        host = parts.hostname.encode("idna").decode("ascii")
    except (ValueError, UnicodeError):
        raise UrlError("Please provide a valid URL.") from None
    if port not in (None, 80, 443):
        raise UrlError("Only standard ports (80 and 443) are allowed.")
    port = port or (443 if parts.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        raise UrlError("Could not resolve the host name.") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.version == 6:
            if ip.sixtofour or ip.teredo:
                raise UrlError("Tunnelled IPv6 addresses are not allowed.")
            if ip.ipv4_mapped:
                ip = ip.ipv4_mapped
        if not ip.is_global:
            raise UrlError("URLs pointing to private or internal addresses are not allowed.")
    return host, infos[0][4][0].split("%")[0], port


def fetch_text(url: str) -> str:
    """Fetch a public web page and return it as plain text.

    Every request, including each redirect, connects to the IP address that was just
    checked, so a redirect or a DNS change can't point the fetch at an internal address.
    HTTPS certificates are still verified against the real hostname.
    """
    with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=False, trust_env=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            host, ip, port = resolve_public(url)
            parts = urlsplit(url)
            netloc = f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}"
            pinned = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
            default_port = port == (443 if parts.scheme == "https" else 80)
            headers = {"Host": host if default_port else f"{host}:{port}", "User-Agent": USER_AGENT}
            extensions = {"sni_hostname": host} if parts.scheme == "https" else {}
            try:
                with client.stream("GET", pinned, headers=headers, extensions=extensions) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers.get("location", ""))
                        continue
                    if response.status_code != 200:
                        raise FetchError(f"The site returned HTTP {response.status_code} instead of the page.")
                    content_type = response.headers.get("content-type", "")
                    if content_type and not content_type.startswith(("text/html", "text/plain", "application/xhtml")):
                        raise FetchError("That URL is not a web page.")
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_BYTES:
                            raise FetchError("The page is larger than 5 MB.")
                    html = bytes(body).decode(response.encoding or "utf-8", errors="replace")
            except httpx.HTTPError as e:
                if "CERTIFICATE" in str(e):
                    raise FetchError("The site's certificate could not be verified.") from None
                raise FetchError("Could not fetch the page. Check the URL and try again.") from None
            converter = html2text.HTML2Text()
            converter.ignore_images = True
            converter.body_width = 0
            return converter.handle(html)
    raise FetchError("The page redirected too many times.")


def friendly_error(e: Exception) -> str:
    if isinstance(e, NoContentError):
        return "No readable content was found at this URL."
    if isinstance(e, (UrlError, FetchError)):
        return str(e)
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
else:
    try:
        with st.spinner("Fetching content and generating summary...", show_time=True):
            llm = GoogleGenAI(model=model, api_key=google_api_key)
            text = fetch_text(url).strip()
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
