# llama-index
<h4 align="center">
  <a href="https://github.com/alphasecio/llama-index/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="Released under the MIT license." />
  </a>
  <a href="https://github.com/alphasecio/llama-index">
    <img src="https://img.shields.io/github/stars/alphasecio/llama-index" alt="GitHub Stars" />
  </a>
  <a href="https://github.com/alphasecio/llama-index">
    <img src="https://img.shields.io/github/forks/alphasecio/llama-index" alt="GitHub Forks" />
  </a>
  <a href="https://github.com/alphasecio/llama-index">
    <img src="https://img.shields.io/github/watchers/alphasecio/llama-index" alt="GitHub Watchers" />
  </a>
  <a href="https://twitter.com/alphasecio">
    <img src="https://img.shields.io/twitter/follow/alphasecio?label=Follow" alt="Follow on Twitter" />
  </a>
</h4>

A collection of Streamlit apps powered by [LlamaIndex](https://www.llamaindex.ai), the open-source framework for building LLM applications over your own data.

| App | What it shows | Keys needed |
|-----|---------------|-------------|
| [chat-with-pdf](./chat-with-pdf) | Parse a PDF with LlamaParse, index it in a `VectorStoreIndex` and ask questions with OpenAI | OpenAI, LlamaCloud |
| [summarize-url](./summarize-url) | Fetch a web page with `SimpleWebPageReader`, build a `SummaryIndex` and summarize it with Google Gemini | Google AI Studio |

API keys are entered in each app's sidebar and kept only in the browser session; nothing is stored server-side.

## Deploy
Both apps deploy together with a one-click [Railway](https://railway.com/deploy/GpZ0J4?referralCode=alphasec) template.

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/GpZ0J4?referralCode=alphasec)

## Run locally
Each app is self-contained. Use a separate virtual environment per app (Python 3.12):

```bash
cd chat-with-pdf        # or summarize-url
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```
