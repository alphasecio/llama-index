# chat-with-csv
A Streamlit app for chatting with CSV files using [LlamaIndex](https://www.llamaindex.ai/). You'll need an API key from [OpenAI](https://platform.openai.com/api-keys) for this project.

Unlike `chat-with-pdf`, this app doesn't require LlamaParse or a LlamaCloud API key — CSV rows are loaded directly with LlamaIndex's `PandasCSVReader` and indexed for semantic search.

> **Note:** this app indexes and retrieves rows semantically, so it's well suited to lookup and summarization questions (e.g. "which rows mention X", "summarize the entries about Y"). It is not a substitute for a SQL/pandas query engine and won't reliably answer precise aggregate questions (e.g. exact sums or averages across the whole file), since it retrieves the most relevant rows rather than computing over all of them.

This app can be deployed on [Railway](https://railway.app/?referralCode=alphasec) like the other apps in this repo. *(A dedicated one-click deploy template/button would need to be created by the maintainer — worth flagging as a follow-up rather than faking the link in this PR.)*