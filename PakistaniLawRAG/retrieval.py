
"""
retrieval.py — Pakistani Law RAG Retrieval Pipeline
Import this module in your Gradio app and evaluation notebook.

Usage:
    from retrieval import load_retrieval_system, retrieve
    load_retrieval_system(pinecone_api_key, chunks_fixed_path, chunks_recursive_path)
    result = retrieve(query, chunk_strategy="fixed", retrieval_mode="hybrid_rerank", top_k=5)
"""

import json, time, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone

# ── Globals (populated by load_retrieval_system) ────────────────
_bi_encoder      = None
_cross_encoder   = None
_pinecone_index  = None
_fixed_chunks    = None
_recursive_chunks = None
_bm25_fixed      = None
_bm25_recursive  = None

NS_FIXED     = "fixed"
NS_RECURSIVE = "recursive"
BM25_TOP_N   = 20
SEMANTIC_TOP_N = 20
RERANK_TOP_N = 20
RRF_K        = 60


def load_retrieval_system(pinecone_api_key: str,
                           index_name: str,
                           fixed_chunks_path: str,
                           recursive_chunks_path: str,
                           device: str = "cpu"):
    """Call once at app startup to initialise all components."""
    global _bi_encoder, _cross_encoder, _pinecone_index
    global _fixed_chunks, _recursive_chunks, _bm25_fixed, _bm25_recursive

    print("Loading bi-encoder...")
    _bi_encoder    = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print("Loading CrossEncoder...")
    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)
    _pinecone_index = pc.Index(index_name)

    print("Loading chunk files and building BM25 indexes...")
    with open(fixed_chunks_path,     "r", encoding="utf-8") as f:
        _fixed_chunks     = json.load(f)
    with open(recursive_chunks_path, "r", encoding="utf-8") as f:
        _recursive_chunks = json.load(f)

    _bm25_fixed     = BM25Okapi([c["text"].lower().split() for c in _fixed_chunks])
    _bm25_recursive = BM25Okapi([c["text"].lower().split() for c in _recursive_chunks])

    print("Retrieval system ready.")


def _bm25_search(query, chunks, bm25_index, top_n=20):
    scores     = bm25_index.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_n]
    results = []
    for rank, idx in enumerate(top_indices):
        r = dict(chunks[idx])
        r["bm25_score"] = float(scores[idx])
        r["bm25_rank"]  = rank + 1
        results.append(r)
    return results


def _semantic_search(query, namespace, top_n=20):
    vec = _bi_encoder.encode(query, convert_to_numpy=True).tolist()
    resp = _pinecone_index.query(vector=vec, top_k=top_n,
                                  namespace=namespace, include_metadata=True)
    results = []
    for rank, m in enumerate(resp.matches):
        results.append({
            "id": m.id,
            "text": m.metadata.get("text", ""),
            "source": m.metadata.get("source", ""),
            "year": m.metadata.get("year", 0),
            "url": m.metadata.get("url", ""),
            "strategy": m.metadata.get("strategy", ""),
            "chunk_idx": m.metadata.get("chunk_idx", 0),
            "semantic_score": float(m.score),
            "semantic_rank": rank + 1,
        })
    return results


def _rrf(bm25_results, semantic_results, k=60, top_n=20):
    all_chunks = {r["id"]: r for r in bm25_results}
    all_chunks.update({r["id"]: r for r in semantic_results})
    bm25_ranks     = {r["id"]: r["bm25_rank"]     for r in bm25_results}
    semantic_ranks = {r["id"]: r["semantic_rank"] for r in semantic_results}
    all_ids = set(bm25_ranks) | set(semantic_ranks)
    rrf_scores = {
        did: (1/(bm25_ranks.get(did, 1e9)+k)) + (1/(semantic_ranks.get(did, 1e9)+k))
        for did in all_ids
    }
    top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_n]
    results = []
    for rank, did in enumerate(top_ids):
        c = dict(all_chunks[did])
        c["rrf_score"] = rrf_scores[did]
        c["rrf_rank"]  = rank + 1
        c["found_by"]  = (["bm25"] if did in bm25_ranks else []) +                           (["semantic"] if did in semantic_ranks else [])
        results.append(c)
    return results


def retrieve(query: str,
             chunk_strategy: str = "fixed",
             retrieval_mode: str = "hybrid_rerank",
             top_k: int = 5) -> dict:
    t0 = time.time()
    latency = {}

    chunks_corpus = _fixed_chunks    if chunk_strategy == "fixed" else _recursive_chunks
    bm25_idx      = _bm25_fixed      if chunk_strategy == "fixed" else _bm25_recursive
    namespace     = NS_FIXED         if chunk_strategy == "fixed" else NS_RECURSIVE

    # BM25
    t1 = time.time()
    bm25_r = _bm25_search(query, chunks_corpus, bm25_idx) if retrieval_mode != "semantic_only" else []
    latency["bm25_ms"] = round((time.time()-t1)*1000, 1)

    # Semantic
    t1 = time.time()
    sem_r = _semantic_search(query, namespace, SEMANTIC_TOP_N)
    latency["semantic_ms"] = round((time.time()-t1)*1000, 1)

    # Fusion
    t1 = time.time()
    fused = _rrf(bm25_r, sem_r) if retrieval_mode != "semantic_only" else sem_r[:RERANK_TOP_N]
    latency["rrf_ms"] = round((time.time()-t1)*1000, 1)

    # Rerank
    t1 = time.time()
    if retrieval_mode == "hybrid_rerank":
        pairs  = [(query, c["text"]) for c in fused]
        scores = _cross_encoder.predict(pairs, show_progress_bar=False)
        for i, c in enumerate(fused): c["crossencoder_score"] = float(scores[i])
        final = sorted(fused, key=lambda x: x["crossencoder_score"], reverse=True)[:top_k]
    else:
        final = fused[:top_k]
    latency["crossencoder_ms"] = round((time.time()-t1)*1000, 1)

    for rank, c in enumerate(final): c["final_rank"] = rank + 1
    latency["total_ms"] = round((time.time()-t0)*1000, 1)

    context = "\n\n".join(
        f"[Source {i+1}: {c['source']} ({c['year']})]'\n'{c['text']}"
        for i, c in enumerate(final)
    )

    return {
        "chunks":   final,
        "context":  context,
        "latency":  latency,
        "metadata": {
            "query": query, "chunk_strategy": chunk_strategy,
            "retrieval_mode": retrieval_mode, "final_returned": len(final)
        }
    }
