# Pakistani Law RAG — Claude Code Setup

## Project Overview
This is a Gradio-based RAG (Retrieval-Augmented Generation) application for Pakistani Law.
It runs locally for testing and deploys to HuggingFace Spaces for production.

## Environment
- Python app, runs with `python app.py`
- `.env` file exists in the project root with all API keys
- Local URL after startup: `http://127.0.0.1:7860`

## Key Files
- `app.py` — main application
- `chunks_fixed.json` — filtered fixed-size chunks (4138 chunks)
- `chunks_recursive.json` — filtered recursive chunks (4768 chunks)
- `requirements.txt` — dependencies
- `.env` — API keys (never commit this)

## Environment Variables (in .env)
- `HF_TOKEN` — HuggingFace access token
- `PINECONE_API_KEY` — Pinecone API key
- `PINECONE_INDEX` — Pinecone index name (pakistani-law)

## Setup Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Verify `.env` exists with all three keys
3. Verify `chunks_fixed.json` and `chunks_recursive.json` exist
4. Run: `python app.py`

## load_dotenv fix
The current `load_dotenv` call looks one directory up. Change it to:
```python
load_dotenv()
```
This finds the `.env` in the current directory.

## Common Issues
- If `HF_TOKEN` not found: check `.env` is in same directory as `app.py`
- If chunks not found: ensure both JSON files are in same directory as `app.py`
- If Pinecone fails: verify API key and index name in `.env`
- Startup takes 60-90 seconds — models loading is normal

## Dependencies
```
gradio>=4.0.0
pinecone
sentence-transformers
rank-bm25
huggingface_hub
numpy
python-dotenv
```
