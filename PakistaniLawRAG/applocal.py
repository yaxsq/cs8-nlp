import os
import time
import json
import re
import numpy as np
import gradio as gr
from groq import Groq
from huggingface_hub import InferenceClient
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
#  CONFIG — reads from environment variables
# ─────────────────────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.environ.get("PINECONE_INDEX", "pakistani-law")
HF_TOKEN         = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")

HF_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
GROQ_MODEL = "llama-3.1-8b-instant"

EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

NS_FIXED      = "fixed"
NS_RECURSIVE  = "recursive"
DEFAULT_TOP_K = 5

# ─────────────────────────────────────────────────────────────
#  GLOBALS (populated at startup)
# ─────────────────────────────────────────────────────────────
bi_encoder       = None
cross_encoder    = None
pinecone_index   = None
fixed_chunks     = None
recursive_chunks = None
bm25_fixed       = None
bm25_recursive   = None
groq_client      = None


def startup():
    """Load all models and data at app startup. Called once."""
    global bi_encoder, cross_encoder, pinecone_index
    global fixed_chunks, recursive_chunks, bm25_fixed, bm25_recursive, groq_client

    print("Loading bi-encoder...")
    bi_encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    print("Loading CrossEncoder...")
    cross_encoder = CrossEncoder(CROSSENCODER_MODEL, device="cpu")

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX)

    print("Loading chunk files...")
    with open("chunks_fixed.json", "r", encoding="utf-8") as f:
        fixed_chunks = json.load(f)
    with open("chunks_recursive.json", "r", encoding="utf-8") as f:
        recursive_chunks = json.load(f)

    print("Building BM25 indexes...")
    bm25_fixed     = BM25Okapi([c["text"].lower().split() for c in fixed_chunks])
    bm25_recursive = BM25Okapi([c["text"].lower().split() for c in recursive_chunks])

    token = os.environ.get("HF_TOKEN", "")
    print(f"HF_TOKEN present: {bool(token)}")
    print(f"HF_TOKEN prefix: {token[:8] if token else 'MISSING'}")

    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq fallback client ready.")
    else:
        print("GROQ_API_KEY not set — fallback unavailable.")

    print("All systems ready.")


# ─────────────────────────────────────────────────────────────
#  LLM CALL — with automatic provider fallback
# ─────────────────────────────────────────────────────────────

def call_llm(prompt: str,
             max_tokens: int = 512,
             temperature: float = 0.1,
             provider: str = "huggingface") -> tuple[str, str]:
    """
    Call LLM with automatic fallback.

    Args:
        prompt: The prompt string
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        provider: Starting provider ("huggingface" or "groq")

    Returns:
        tuple: (response_text, active_provider)
        active_provider may differ from input provider if fallback occurred.
    """
    providers_to_try = (
        ["huggingface", "groq"] if provider == "huggingface"
        else ["groq", "huggingface"]
    )

    for current_provider in providers_to_try:
        try:
            if current_provider == "huggingface":
                hf_client = InferenceClient(token=HF_TOKEN)
                response = hf_client.chat_completion(
                    model=HF_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                )
                return response.choices[0].message.content.strip(), "huggingface"

            elif current_provider == "groq" and groq_client:
                response = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                )
                return response.choices[0].message.content.strip(), "groq"

        except Exception as e:
            err = str(e)
            print(f"  [{current_provider}] failed: {err[:100]}")
            if current_provider == providers_to_try[0] and len(providers_to_try) > 1:
                print(f"  Falling back to {providers_to_try[1]}...")
            time.sleep(1)

    return "[LLM Error: All providers failed]", provider


# ─────────────────────────────────────────────────────────────
#  QUERY REWRITING
# ─────────────────────────────────────────────────────────────

def rewrite_query_for_legal(query: str,
                             provider: str = "huggingface") -> tuple[str, str]:
    prompt = f"""You are a Pakistani legal expert.
Rewrite this question using precise Pakistani legal terminology.
Include both English terms AND specific legal terms used in Pakistani law.
Return ONLY the rewritten query terms. No explanation. No prefix.
Maximum 20 words.

Examples:
Input: murder punishment
Output: qatl-i-amd punishment death imprisonment qisas diyat Section 302 PPC

Input: theft penalty
Output: theft dishonestly takes property Section 378 379 PPC

Input: terrorism definition
Output: terrorist act scheduled offence Anti-Terrorism Act 1997

Input: fundamental rights
Output: fundamental rights Article 8 9 10 25 Constitution Pakistan

Input: {query}
Output:"""
    try:
        rewritten, active_provider = call_llm(
            prompt, max_tokens=40, temperature=0.0, provider=provider
        )
        rewritten = re.sub(
            r'^(rewritten|query|output|answer|here|terms?|result)[:\s]+',
            '', rewritten, flags=re.IGNORECASE
        ).strip()
        if not rewritten or len(rewritten) > 200:
            return query, active_provider
        return rewritten, active_provider
    except Exception:
        return query, provider


# ─────────────────────────────────────────────────────────────
#  RETRIEVAL
# ─────────────────────────────────────────────────────────────

def bm25_search(query, chunks, bm25_index, top_n=20):
    scores      = bm25_index.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_n]
    results = []
    for rank, idx in enumerate(top_indices):
        r = dict(chunks[idx])
        r["bm25_score"] = float(scores[idx])
        r["bm25_rank"]  = rank + 1
        results.append(r)
    return results


def semantic_search(query, namespace, top_n=20):
    vec  = bi_encoder.encode(query, convert_to_numpy=True).tolist()
    resp = pinecone_index.query(vector=vec, top_k=top_n,
                                 namespace=namespace, include_metadata=True)
    results = []
    for rank, m in enumerate(resp.matches):
        results.append({
            "id":             m.id,
            "text":           m.metadata.get("text", ""),
            "source":         m.metadata.get("source", ""),
            "year":           m.metadata.get("year", ""),
            "url":            m.metadata.get("url", ""),
            "strategy":       m.metadata.get("strategy", ""),
            "chunk_idx":      m.metadata.get("chunk_idx", 0),
            "semantic_score": float(m.score),
            "semantic_rank":  rank + 1,
        })
    return results


def rrf_fusion(bm25_results, semantic_results, k=60, top_n=20):
    all_chunks     = {r["id"]: r for r in bm25_results}
    all_chunks.update({r["id"]: r for r in semantic_results})
    bm25_ranks     = {r["id"]: r["bm25_rank"]     for r in bm25_results}
    semantic_ranks = {r["id"]: r["semantic_rank"] for r in semantic_results}
    all_ids = set(bm25_ranks) | set(semantic_ranks)
    scores  = {
        did: (1 / (bm25_ranks.get(did, 1e9) + k)) +
             (1 / (semantic_ranks.get(did, 1e9) + k))
        for did in all_ids
    }
    top_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
    results = []
    for rank, did in enumerate(top_ids):
        c = dict(all_chunks[did])
        c["rrf_score"] = scores[did]
        c["rrf_rank"]  = rank + 1
        c["found_by"]  = (["bm25"] if did in bm25_ranks else []) + \
                          (["semantic"] if did in semantic_ranks else [])
        results.append(c)
    return results


def retrieve(query, chunk_strategy="recursive",
             retrieval_mode="hybrid_rerank", top_k=5,
             provider="huggingface") -> tuple[dict, str]:
    t0 = time.time()

    retrieval_query, active_provider = rewrite_query_for_legal(
        query, provider=provider
    )

    chunks_corpus = fixed_chunks    if chunk_strategy == "fixed" else recursive_chunks
    bm25_idx      = bm25_fixed      if chunk_strategy == "fixed" else bm25_recursive
    namespace     = NS_FIXED        if chunk_strategy == "fixed" else NS_RECURSIVE

    bm25_r = bm25_search(retrieval_query, chunks_corpus, bm25_idx) \
             if retrieval_mode != "semantic_only" else []
    sem_r  = semantic_search(retrieval_query, namespace, 20)

    if retrieval_mode != "semantic_only":
        fused = rrf_fusion(bm25_r, sem_r)
    else:
        fused = sem_r[:20]

    if retrieval_mode == "hybrid_rerank" and fused:
        pairs  = [(retrieval_query, c["text"]) for c in fused]
        scores = cross_encoder.predict(pairs, show_progress_bar=False)
        for i, c in enumerate(fused):
            c["crossencoder_score"] = float(scores[i])
        final = sorted(fused, key=lambda x: x["crossencoder_score"],
                       reverse=True)[:top_k]
    else:
        final = fused[:top_k]

    for rank, c in enumerate(final):
        c["final_rank"] = rank + 1

    context_parts = [
        f"[Source {i+1}: {c['source']} ({c['year']})]\n{c['text']}"
        for i, c in enumerate(final)
    ]
    context = "\n\n".join(context_parts)

    return {
        "chunks":     final,
        "context":    context,
        "latency_ms": round((time.time() - t0) * 1000, 1)
    }, active_provider


# ─────────────────────────────────────────────────────────────
#  GENERATION
# ─────────────────────────────────────────────────────────────

def build_generation_prompt(query: str, context: str) -> str:
    return f"""You are a legal assistant for Pakistani law.
Answer the question using the context provided below.
Extract and combine relevant information from all sources.
Be specific and cite section numbers when mentioned in the context.
Only if the context contains absolutely no relevant information say:
"The provided legal documents do not contain sufficient information to answer this question."

Context:
{context}

Question: {query}

Answer:"""

def generate_answer(query: str, context: str,
                    provider: str = "huggingface") -> tuple[str, str]:
    prompt = build_generation_prompt(query, context)
    return call_llm(prompt, max_tokens=600, provider=provider)


# ─────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────

def extract_claims(answer: str,
                   provider: str = "huggingface") -> tuple[list, str]:
    prompt = f"""You are a precise claim extractor.
Extract every distinct factual claim from the following text.
Return ONLY a numbered list, one claim per line.
Each claim must be a complete standalone sentence.

Text:
{answer}

Numbered list of factual claims:"""
    response, active_provider = call_llm(
        prompt, max_tokens=400, temperature=0.0, provider=provider
    )
    claims = []
    for line in response.strip().split("\n"):
        match = re.match(r"^\d+[.)\s]+(.+)$", line.strip())
        if match and len(match.group(1).strip()) > 10:
            claims.append(match.group(1).strip())
    return claims, active_provider


def verify_claim(claim: str, context: str,
                 provider: str = "huggingface") -> tuple[dict, str]:
    prompt = f"""You are a fact-checker for legal documents.
Determine if the following claim is supported by the provided context.
Answer with ONLY 'YES' or 'NO' on the first line, then one sentence of reasoning.

Context:
{context[:2000]}

Claim: {claim}

Is this claim supported by the context?"""
    response, active_provider = call_llm(
        prompt, max_tokens=100, temperature=0.0, provider=provider
    )
    first_line = response.strip().split("\n")[0].strip().upper()
    supported  = "YES" in first_line and "NO" not in first_line[:3]
    return {"claim": claim, "supported": supported,
            "reasoning": response.strip()}, active_provider


def compute_faithfulness(answer: str, context: str,
                         provider: str = "huggingface") -> tuple[dict, str]:
    claims, active_provider = extract_claims(answer, provider=provider)
    if not claims:
        return {"score": 0.0, "claims": [], "n_claims": 0, "n_supported": 0}, active_provider
    verifications = []
    for claim in claims:
        result, active_provider = verify_claim(claim, context, provider=active_provider)
        verifications.append(result)
    n_supported = sum(1 for v in verifications if v["supported"])
    return {
        "score":       round(n_supported / len(verifications), 4),
        "claims":      verifications,
        "n_claims":    len(verifications),
        "n_supported": n_supported
    }, active_provider


def compute_relevancy(query: str, answer: str,
                      provider: str = "huggingface") -> tuple[dict, str]:
    prompt = f"""You are given an answer to a legal question about Pakistani law.
Generate exactly 3 different questions that this answer could be responding to.
Return ONLY the 3 questions as a numbered list. Nothing else.

Answer:
{answer}

3 questions:"""
    response, active_provider = call_llm(
        prompt, max_tokens=200, temperature=0.3, provider=provider
    )
    questions = []
    for line in response.strip().split("\n"):
        match = re.match(r"^\d+[.)\s]+(.+\?)$", line.strip())
        if match and len(match.group(1)) > 10:
            questions.append(match.group(1).strip())
        elif line.strip().endswith("?") and len(line.strip()) > 10:
            questions.append(line.strip())
    questions = questions[:3]
    if not questions:
        return {"score": 0.0, "generated_questions": [],
                "similarities": []}, active_provider

    query_vec = bi_encoder.encode(query,     convert_to_numpy=True)
    gen_vecs  = bi_encoder.encode(questions, convert_to_numpy=True)
    sims = []
    for gv in gen_vecs:
        q_n = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        g_n = gv        / (np.linalg.norm(gv)        + 1e-10)
        sims.append(round(float(np.dot(q_n, g_n)), 4))
    return {
        "score":               round(float(np.mean(sims)), 4),
        "generated_questions": questions,
        "similarities":        sims
    }, active_provider


# ─────────────────────────────────────────────────────────────
#  MAIN PIPELINE FUNCTION (called by Gradio)
# ─────────────────────────────────────────────────────────────

def run_pipeline(query, chunk_strategy, retrieval_mode,
                 show_evaluation, provider):
    if not query.strip():
        return "", "", "", "", provider

    # 1. Retrieve (may trigger fallback inside)
    retrieval, active_provider = retrieve(
        query,
        chunk_strategy=chunk_strategy,
        retrieval_mode=retrieval_mode,
        top_k=DEFAULT_TOP_K,
        provider=provider
    )

    # 2. Generate
    t0 = time.time()
    answer, active_provider = generate_answer(
        query, retrieval["context"], provider=active_provider
    )
    gen_ms = round((time.time() - t0) * 1000, 1)

    # 3. Format retrieved context for display
    context_display = ""
    for i, chunk in enumerate(retrieval["chunks"]):
        source    = chunk.get("source", "Unknown")
        year      = chunk.get("year", "")
        text      = chunk.get("text", "")
        found     = ", ".join(chunk.get("found_by", []))
        ce        = chunk.get("crossencoder_score", None)
        score_str = f"CE={ce:.3f}" if ce is not None else \
                    f"RRF={chunk.get('rrf_score', 0):.4f}"
        context_display += (
            f"**[{i+1}] {source} ({year})** — {score_str}"
            + (f" | found_by: {found}" if found else "")
            + f"\n\n{text}\n\n---\n\n"
        )

    # 4. Provider indicator in latency line
    provider_label = (
        "🤗 HuggingFace" if active_provider == "huggingface"
        else "⚡ Groq (fallback)"
    )
    latency_str = (
        f"**Provider:** {provider_label}  |  "
        f"**Retrieval:** {retrieval['latency_ms']}ms  |  "
        f"**Generation:** {gen_ms}ms  |  "
        f"**Total:** {retrieval['latency_ms'] + gen_ms}ms"
    )

    # 5. Evaluation (optional — costs extra LLM calls)
    eval_display = ""
    if show_evaluation:
        faith, active_provider = compute_faithfulness(
            answer, retrieval["context"], provider=active_provider
        )
        relev, active_provider = compute_relevancy(
            query, answer, provider=active_provider
        )
        eval_display += f"### Faithfulness Score: {faith['score']:.2%}\n"
        eval_display += (f"*{faith['n_supported']}/{faith['n_claims']}"
                         f" claims supported by context*\n\n")
        for v in faith["claims"]:
            icon = "✅" if v["supported"] else "❌"
            eval_display += f"{icon} {v['claim']}\n\n"
        eval_display += f"---\n\n### Relevancy Score: {relev['score']:.4f}\n"
        eval_display += "*Generated questions from answer vs original query:*\n\n"
        for q, sim in zip(relev["generated_questions"], relev["similarities"]):
            eval_display += f"- [{sim:.4f}] {q}\n"

    # Return active_provider as 5th output so Gradio updates the dropdown
    return answer, context_display, latency_str, eval_display, active_provider


# ─────────────────────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Pakistani Law RAG", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🏛️ Pakistani Law — RAG Question Answering System
    Ask any question about Pakistani law. The system retrieves relevant legal text
    from 10 acts including the Constitution, PPC, Contract Act, NAO, and more.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What is the punishment for murder under Pakistani law?",
                lines=2
            )
        with gr.Column(scale=1):
            chunk_strategy = gr.Dropdown(
                choices=["recursive", "fixed"],
                value="recursive",
                label="Chunking Strategy"
            )
            retrieval_mode = gr.Dropdown(
                choices=["hybrid_rerank", "semantic_only"],
                value="hybrid_rerank",
                label="Retrieval Mode"
            )
            provider_select = gr.Dropdown(
                choices=["huggingface", "groq"],
                value="huggingface",
                label="LLM Provider (auto-switches on failure)"
            )
            show_eval = gr.Checkbox(
                value=False,
                label="Run Faithfulness & Relevancy Evaluation"
            )
            submit_btn = gr.Button("Ask", variant="primary")

    gr.Markdown(
        "*HuggingFace is used by default. "
        "If its limit is reached, the system automatically switches to Groq "
        "and the dropdown above will update to reflect this.*"
    )

    with gr.Row():
        answer_output = gr.Markdown(label="Answer")

    latency_output = gr.Markdown()

    with gr.Accordion("Retrieved Context", open=False):
        context_output = gr.Markdown()

    with gr.Accordion("Evaluation Results", open=False):
        eval_output = gr.Markdown()

    gr.Examples(
        examples=[
            ["What is the punishment for murder under the Pakistan Penal Code?",
             "recursive", "hybrid_rerank", False, "huggingface"],
            ["What fundamental rights does the Constitution of Pakistan guarantee?",
             "recursive", "hybrid_rerank", False, "huggingface"],
            ["What are the essential elements of a valid contract in Pakistan?",
             "recursive", "hybrid_rerank", False, "huggingface"],
            ["What offences fall under the National Accountability Ordinance?",
             "recursive", "hybrid_rerank", False, "huggingface"],
            ["What is the procedure for filing a First Information Report?",
             "recursive", "hybrid_rerank", False, "huggingface"],
        ],
        inputs=[query_input, chunk_strategy, retrieval_mode,
                show_eval, provider_select],
    )

    submit_btn.click(
        fn=run_pipeline,
        inputs=[query_input, chunk_strategy, retrieval_mode,
                show_eval, provider_select],
        outputs=[answer_output, context_output, latency_output,
                 eval_output, provider_select]
    )
    query_input.submit(
        fn=run_pipeline,
        inputs=[query_input, chunk_strategy, retrieval_mode,
                show_eval, provider_select],
        outputs=[answer_output, context_output, latency_output,
                 eval_output, provider_select]
    )


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    startup()
    demo.launch(ssr_mode=False)
