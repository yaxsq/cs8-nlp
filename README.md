# Pakistani Law RAG — Question Answering System

A Retrieval-Augmented Generation (RAG) application for Pakistani law, built with Gradio. Retrieves relevant legal text from 10 Pakistani acts and generates grounded answers using an LLM.

## Live Demo

Deployed on HuggingFace Spaces — runs `app.py` automatically.

## Features

- **Hybrid retrieval**: BM25 + semantic search fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks top candidates
- **Legal query rewriting**: LLM rewrites queries into Pakistani legal terminology before retrieval
- **Dual LLM provider with automatic fallback**: HuggingFace Inference API is primary; Groq auto-activates on any failure (402, 410, 429, etc.) and the UI dropdown updates to reflect the switch
- **Optional RAG evaluation**: Faithfulness (claim verification) and Answer Relevancy scoring

## Models

| Role | Model |
|---|---|
| Generation | `meta-llama/Llama-3.1-8B-Instruct` (HF primary) |
| Fallback | `llama-3.1-8b-instant` via Groq |
| Embeddings | `all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

## Legal Corpus

10 Pakistani acts including:
- Constitution of Pakistan
- Pakistan Penal Code (PPC)
- Contract Act
- National Accountability Ordinance (NAO)
- Code of Criminal Procedure (CrPC)
- Anti-Terrorism Act
- and more

Chunked into two strategies:
- `chunks_fixed.json` — fixed-size chunks (4138 chunks)
- `chunks_recursive.json` — recursive chunks (4768 chunks)

## Files

```
app.py              — HuggingFace Spaces entry point
applocal.py         — Local development entry point
requirements.txt    — Python dependencies
CLAUDE.md           — Claude Code setup notes
chunks_fixed.json   — Fixed-size chunk corpus (not in repo — add manually)
chunks_recursive.json — Recursive chunk corpus (not in repo — add manually)
```

## Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env in PakistaniLawRAG/ with:
#    HF_TOKEN=...
#    PINECONE_API_KEY=...
#    PINECONE_INDEX=pakistani-law
#    GROQ_API_KEY=...

# 3. Place chunks_fixed.json and chunks_recursive.json in PakistaniLawRAG/

# 4. Run
python applocal.py
```

App starts at `http://127.0.0.1:7860`

## Environment Variables

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace access token (primary LLM) |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX` | Pinecone index name (`pakistani-law`) |
| `GROQ_API_KEY` | Groq API key (fallback LLM) |

## Retrieval Pipeline

1. Query is rewritten using Pakistani legal terminology (LLM)
2. BM25 search over local chunk corpus
3. Semantic search via Pinecone vector index
4. RRF fusion of BM25 + semantic results
5. Cross-encoder reranking of top-20 fused results
6. Top-5 chunks passed to LLM for answer generation
