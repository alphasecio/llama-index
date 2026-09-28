# summarize-url
A Streamlit app for summarizing web pages using [LlamaIndex](https://www.llamaindex.ai) and [Google Gemini](https://ai.google.dev/gemini-api/docs). You'll need an [API key](https://aistudio.google.com/app/apikey) from Google AI Studio for this project.

![summarize-url](./summarize-url.png)

## How it works
1. The page is fetched and converted to text with LlamaIndex's `SimpleWebPageReader`.
2. The text is loaded into a `SummaryIndex`.
3. A Gemini model produces a 200-250 word summary using the `tree_summarize` response mode.

Only public `http`/`https` URLs on standard ports are accepted; private and internal addresses are blocked. Very long pages are truncated before summarizing to keep Gemini usage in check.

## Configuration
All settings are optional environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_MODEL` | `gemini-3.7-flash` | Default Gemini model (editable in the sidebar) |
| `MAX_CHARS` | `60000` | Maximum characters of page text sent for summarization |

## Run and deploy
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

For a detailed guide, see [this](https://alphasec.io/blinkist-for-urls-with-llama-index-and-openai) post. To deploy on [Railway](https://railway.com?referralCode=alphasec) using a one-click template, click the button below.

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/GpZ0J4?referralCode=alphasec)
