import logging
import os

import openai
import streamlit as st
from llama_cloud import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    LlamaCloud,
    PollingError,
    PollingTimeoutError,
)
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


def env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, default))
        return value if value > 0 else default
    except ValueError:
        return default


# Defaults can be overridden with environment variables
PARSE_TIERS = ["fast", "cost_effective", "agentic", "agentic_plus"]
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
DEFAULT_TIER = os.getenv("LLAMA_PARSE_TIER", "cost_effective").strip()
MAX_PAGES = env_int("MAX_PAGES", 50)
MAX_UPLOAD_MB = 20  # keep in sync with server.maxUploadSize in .streamlit/config.toml
TOP_K = 5


class NoContentError(Exception):
    """Raised when parsing returns no usable text."""


def parse_pdf(data: bytes, name: str, api_key: str, tier: str) -> list[Document]:
    """Parse a PDF with LlamaParse and return one LlamaIndex Document per page."""
    with LlamaCloud(api_key=api_key) as client:
        result = client.parsing.parse(
            tier=tier,
            version="latest",
            upload_file=(name, data, "application/pdf"),
            page_ranges={"max_pages": MAX_PAGES},
            expand=["markdown"],
        )
    pages = result.markdown.pages if result.markdown else []
    return [
        Document(text=page.markdown, metadata={"page": page.page_number})
        for page in pages
        if page.success and page.markdown.strip()
    ]


def friendly_error(e: Exception) -> str:
    if isinstance(e, NoContentError):
        return "No text could be extracted from this PDF."
    if isinstance(e, AuthenticationError):
        return "LlamaCloud rejected the API key."
    if isinstance(e, PollingTimeoutError):
        return "LlamaParse timed out. Try a smaller document or a faster tier."
    if isinstance(e, PollingError):
        return "LlamaParse could not parse this document."
    if isinstance(e, APIStatusError):
        return f"LlamaCloud returned an error (HTTP {e.status_code})."
    if isinstance(e, APIConnectionError):
        return "Could not reach LlamaCloud. Please try again."
    if isinstance(e, openai.AuthenticationError):
        return "OpenAI rejected the API key."
    if isinstance(e, openai.NotFoundError):
        return "The OpenAI model was not found, or your key has no access to it."
    if isinstance(e, openai.RateLimitError):
        return "OpenAI rate limit or quota exceeded."
    if isinstance(e, openai.APIError):
        return "OpenAI returned an error. Please try again."
    if isinstance(e, ValueError) and "Unknown model" in str(e):
        return "Unknown OpenAI model name."
    return f"Unexpected error ({type(e).__name__})."


# Streamlit app config
st.set_page_config(page_title="Chat with PDF", page_icon="📄")

with st.sidebar:
    st.header("Settings")
    st.text_input("OpenAI API key", type="password", key="openai_api_key")
    st.text_input("LlamaCloud API key", type="password", key="llama_cloud_api_key")
    st.caption(
        "Get keys from [OpenAI](https://platform.openai.com/api-keys) and "
        "[LlamaCloud](https://cloud.llamaindex.ai). Keys stay in your browser session only."
    )
    st.divider()
    st.text_input("OpenAI model", value=DEFAULT_MODEL, key="openai_model")
    st.selectbox(
        "LlamaParse tier",
        PARSE_TIERS,
        index=PARSE_TIERS.index(DEFAULT_TIER) if DEFAULT_TIER in PARSE_TIERS else 1,
        key="parse_tier",
        help="Higher tiers handle complex layouts better but use more credits.",
    )

st.header("📄 Chat with PDF")
st.caption("Parse with LlamaParse, index with LlamaIndex, answer with OpenAI.")

source_doc = st.file_uploader(
    f"Upload a PDF (up to {MAX_UPLOAD_MB} MB, first {MAX_PAGES} pages are parsed)",
    type="pdf",
    max_upload_size=MAX_UPLOAD_MB,
)

with st.form("query_form", border=False):
    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
    query = col1.text_input(
        "Query",
        placeholder="Ask a question about your PDF...",
        label_visibility="collapsed",
        max_chars=500,
    )
    submit = col2.form_submit_button("Ask", type="primary", width="stretch")

if not submit:
    if not source_doc:
        st.info("Upload a PDF, add your API keys in the sidebar, then ask a question.")
    st.stop()

openai_api_key = st.session_state.openai_api_key.strip()
llama_cloud_api_key = st.session_state.llama_cloud_api_key.strip()
model = st.session_state.openai_model.strip()
tier = st.session_state.parse_tier

if not openai_api_key:
    st.error("Please provide the OpenAI API key.")
elif not llama_cloud_api_key:
    st.error("Please provide the LlamaCloud API key.")
elif not model:
    st.error("Please provide an OpenAI model name.")
elif not source_doc:
    st.error("Please upload a PDF document.")
elif not query.strip():
    st.error("Please provide a query.")
else:
    if source_doc.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"The PDF is larger than {MAX_UPLOAD_MB} MB.")
        st.stop()
    data = source_doc.getvalue()
    if b"%PDF-" not in data[:1024]:
        st.error("The uploaded file does not look like a valid PDF.")
        st.stop()

    # Parse once per document and tier; re-index only when the document, tier, or embedding model changes
    doc_key = (source_doc.file_id, tier)
    index_key = (doc_key, openai_api_key, EMBED_MODEL)

    if st.session_state.get("index_key") != index_key:
        with st.status("Preparing document...", expanded=False) as status:
            try:
                if st.session_state.get("doc_key") != doc_key:
                    status.update(label=f"Parsing with LlamaParse ({tier})...")
                    st.session_state.pop("documents", None)
                    st.session_state.pop("doc_key", None)
                    documents = parse_pdf(data, source_doc.name, llama_cloud_api_key, tier)
                    if not documents:
                        raise NoContentError()
                    st.session_state.documents = documents
                    st.session_state.doc_key = doc_key

                status.update(label="Building vector index...")
                st.session_state.pop("index", None)
                st.session_state.pop("index_key", None)
                st.session_state.index = VectorStoreIndex.from_documents(
                    st.session_state.documents,
                    embed_model=OpenAIEmbedding(model=EMBED_MODEL, api_key=openai_api_key),
                )
                st.session_state.index_key = index_key
                pages = len(st.session_state.documents)
                status.update(label=f"Indexed {pages} page(s).", state="complete")
            except Exception as e:
                logger.exception("Indexing failed")
                status.update(label="Indexing failed.", state="error")
                st.error(friendly_error(e))
                st.stop()

    with st.spinner("Thinking...", show_time=True):
        try:
            query_engine = st.session_state.index.as_query_engine(
                llm=OpenAI(model=model, api_key=openai_api_key),
                similarity_top_k=TOP_K,
                response_mode="tree_summarize",
            )
            response = query_engine.query(query)
        except Exception as e:
            logger.exception("Query failed")
            st.error(friendly_error(e))
            st.stop()

    with st.container(border=True):
        st.markdown("**Answer**")
        st.markdown(response.response or "_No answer returned._")
        sources = sorted({n.node.metadata.get("page") for n in response.source_nodes} - {None})
        if sources:
            label = "page" if len(sources) == 1 else "pages"
            st.caption(f"Sources: {label} " + ", ".join(str(p) for p in sources))

    if response.source_nodes:
        with st.expander("Retrieved passages"):
            for n in response.source_nodes:
                score = f" · score {n.score:.2f}" if n.score is not None else ""
                st.caption(f"Page {n.node.metadata.get('page', '?')}{score}")
                st.text(n.node.get_content()[:800])
