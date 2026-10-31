# Insurellm RAG

Week 5 knowledge worker, stripped down: load markdown, cut chunks, embed, pick closest, ask the model. Gradio chat on the side.

## Run

```bash
cd insurellm-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your OPENAI_API_KEY in .env
python check.py
python app.py
```

First `app.py` run embeds the knowledge base and writes `data/store.pkl`. Later runs reuse that file. Delete `data/` to rebuild.

## Layout

| File | Job |
|---|---|
| `knowledge-base/` | Insurellm markdown |
| `knowledge.py` | load, chunk, find closest |
| `openai_client.py` | embeddings + chat |
| `app.py` | Gradio: chat + retrieved context |
| `check.py` | ranking self-check (no API) |
