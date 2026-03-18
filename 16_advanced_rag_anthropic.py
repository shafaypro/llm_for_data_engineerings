"""
16_advanced_rag_anthropic.py
=============================
Concept: Advanced RAG Patterns
--------------------------------
Beyond basic RAG: techniques that dramatically improve retrieval quality
and answer accuracy for production data engineering use cases.

Techniques covered:
1. Hybrid search (keyword + semantic)
2. Re-ranking retrieved chunks
3. Query decomposition (complex questions → sub-queries)
4. HyDE — Hypothetical Document Embedding
5. Self-querying with metadata filters
6. Contextual chunk headers (improves embedding quality)

Install: pip install anthropic voyageai numpy
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
          export VOYAGE_API_KEY=pa-...
"""

import anthropic
import json
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

client = anthropic.Anthropic()

try:
    import voyageai
    vo = voyageai.Client()
    def _embed(texts: List[str]) -> List[np.ndarray]:
        result = vo.embed(texts, model="voyage-3-lite")
        return [np.array(e) for e in result.embeddings]
except ImportError:
    def _embed(texts: List[str]) -> List[np.ndarray]:
        return [np.random.RandomState(hash(t) % 2**32).randn(512) for t in texts]


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def cosine_sim(self, query_emb: np.ndarray) -> float:
        if self.embedding is None:
            return 0.0
        return float(np.dot(self.embedding, query_emb) /
                     (np.linalg.norm(self.embedding) * np.linalg.norm(query_emb) + 1e-9))


# Sample knowledge base
DOCUMENTS = [
    {
        "text": "The daily_orders_etl DAG runs at 2am UTC. It extracts from MySQL, transforms in Python, loads to Snowflake. SLA is 4am UTC. On failure, first check the RDS replica lag. Common fix: increase connection pool size in airflow.cfg.",
        "source": "ops_runbook.md",
        "metadata": {"type": "runbook", "topic": "airflow", "dag": "daily_orders_etl"}
    },
    {
        "text": "fct_orders dbt model: grain=order_id, refreshed daily, depends on stg_orders+dim_customers. Known issue: orders before 2022 have null product_id. Row count ~5M. Partition key: order_date.",
        "source": "dbt_docs.md",
        "metadata": {"type": "dbt_docs", "topic": "models", "model": "fct_orders"}
    },
    {
        "text": "Snowflake cost optimisation: use SMALL warehouse for dbt runs <10min, MEDIUM for full refreshes. Enable auto-suspend after 2 minutes. Avoid SELECT * on large tables. Use COPY INTO for bulk loads, not INSERT.",
        "source": "snowflake_guide.md",
        "metadata": {"type": "guide", "topic": "snowflake", "subtopic": "cost"}
    },
    {
        "text": "Data quality monitoring: run dbt tests after each model. Alert thresholds: null rate >5% = P2, >15% = P1. Duplicate rate >1% = P2. Use elementary-data for automated anomaly detection.",
        "source": "data_quality.md",
        "metadata": {"type": "guide", "topic": "data_quality"}
    },
    {
        "text": "On-call rotation: Alice (week 1), Bob (week 2), Carlos (week 3). Escalation: Slack #data-alerts → PagerDuty → Engineering Manager. P1 response: 15 minutes. P2 response: 2 hours.",
        "source": "oncall.md",
        "metadata": {"type": "process", "topic": "oncall"}
    },
]


def build_store(docs: list) -> List[Chunk]:
    """Embed all documents and return as chunks."""
    chunks = [Chunk(text=d["text"], source=d["source"], metadata=d["metadata"]) for d in docs]
    embeddings = _embed([c.text for c in chunks])
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    return chunks


# ── Technique 1: Hybrid search (BM25-style keyword + semantic) ────────────────

def technique_1_hybrid_search(store: List[Chunk], query: str, top_k: int = 3) -> List[Chunk]:
    """
    Combine keyword overlap score with semantic similarity.
    Keyword search catches exact matches (table names, DAG IDs) that
    semantic search sometimes misses because they're rare tokens.
    """
    query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())

    def keyword_score(text: str) -> float:
        text_words = set(re.sub(r'[^\w\s]', '', text.lower()).split())
        overlap = len(query_words & text_words)
        return overlap / (len(query_words) + 1)

    q_emb = _embed([query])[0]
    scored = []
    for chunk in store:
        semantic = chunk.cosine_sim(q_emb)
        keyword = keyword_score(chunk.text)
        # Weighted combination: 70% semantic, 30% keyword
        combined = 0.7 * semantic + 0.3 * keyword
        scored.append((combined, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ── Technique 2: Re-ranking ────────────────────────────────────────────────────

def technique_2_reranking(store: List[Chunk], query: str, initial_k: int = 5, final_k: int = 2) -> List[Chunk]:
    """
    Retrieve more candidates than needed (initial_k), then use Claude to
    re-rank them by actual relevance to the question. More expensive but
    much more accurate — catches semantic search misses.
    """
    # Initial broad retrieval
    q_emb = _embed([query])[0]
    candidates = sorted(store, key=lambda c: c.cosine_sim(q_emb), reverse=True)[:initial_k]

    # Ask Claude to re-rank
    candidates_str = "\n\n".join([
        f"[Chunk {i+1}]\n{c.text}"
        for i, c in enumerate(candidates)
    ])

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        temperature=0,
        system="Re-rank document chunks by relevance to the query. Return ONLY a JSON array of chunk numbers in order of relevance, e.g. [3, 1, 2]",
        messages=[{"role": "user", "content": f"Query: {query}\n\nChunks:\n{candidates_str}"}]
    )

    try:
        ranking = json.loads(response.content[0].text.strip())
        reranked = [candidates[i-1] for i in ranking if 1 <= i <= len(candidates)]
        return reranked[:final_k]
    except Exception:
        return candidates[:final_k]


# ── Technique 3: Query decomposition ─────────────────────────────────────────

def technique_3_query_decomposition(store: List[Chunk], complex_query: str) -> str:
    """
    Break a complex multi-part question into sub-queries, answer each separately,
    then synthesise. Much better than trying to answer everything at once.
    """
    # Step 1: Decompose
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        system='Break complex questions into 2-4 simpler sub-questions. Return JSON: {"sub_questions": ["q1", "q2"]}',
        messages=[{"role": "user", "content": f"Decompose: {complex_query}"}]
    )

    try:
        sub_questions = json.loads(response.content[0].text.strip())["sub_questions"]
    except Exception:
        sub_questions = [complex_query]

    print(f"  Decomposed into {len(sub_questions)} sub-questions:")
    for q in sub_questions:
        print(f"    - {q}")

    # Step 2: Answer each sub-question
    sub_answers = []
    for sub_q in sub_questions:
        chunks = technique_1_hybrid_search(store, sub_q, top_k=2)
        context = "\n".join([c.text for c in chunks])
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system="Answer using only the context provided. Be concise.",
            messages=[{"role": "user", "content": f"<context>{context}</context>\n\nQuestion: {sub_q}"}]
        )
        sub_answers.append(f"Q: {sub_q}\nA: {resp.content[0].text.strip()}")

    # Step 3: Synthesise
    synthesis_context = "\n\n".join(sub_answers)
    final = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=250,
        system="Synthesise sub-answers into one coherent response.",
        messages=[{"role": "user", "content": f"<sub_answers>\n{synthesis_context}\n</sub_answers>\n\nOriginal question: {complex_query}"}]
    )
    return final.content[0].text.strip()


# ── Technique 4: HyDE (Hypothetical Document Embedding) ──────────────────────

def technique_4_hyde(store: List[Chunk], query: str, top_k: int = 3) -> List[Chunk]:
    """
    HyDE: instead of embedding the question, generate a hypothetical
    answer document and embed THAT. The hypothetical answer is closer in
    embedding space to real answer documents than the raw question is.

    Particularly good for technical questions where the question and answer
    use very different vocabulary.
    """
    # Generate hypothetical document
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        system="Generate a short, realistic documentation snippet that would answer the following question. Write it as if it's from a real runbook or technical doc, not as an answer.",
        messages=[{"role": "user", "content": query}]
    )
    hypothetical_doc = response.content[0].text.strip()

    # Embed the hypothetical document instead of the raw query
    hyp_emb = _embed([hypothetical_doc])[0]
    scored = sorted(store, key=lambda c: c.cosine_sim(hyp_emb), reverse=True)
    return scored[:top_k]


# ── Technique 5: Metadata filtering ──────────────────────────────────────────

def technique_5_metadata_filtering(store: List[Chunk], query: str, filters: dict, top_k: int = 3) -> List[Chunk]:
    """
    Filter chunks by metadata BEFORE doing semantic search.
    Much more precise when you know what type of doc you need.
    E.g. only search runbooks when diagnosing an incident.
    """
    filtered = [c for c in store if all(c.metadata.get(k) == v for k, v in filters.items())]

    if not filtered:
        print(f"  No chunks matched filters {filters}, falling back to full search")
        filtered = store

    q_emb = _embed([query])[0]
    scored = sorted(filtered, key=lambda c: c.cosine_sim(q_emb), reverse=True)
    return scored[:top_k]


# ── Demo: compare basic RAG vs advanced techniques ───────────────────────────

def main():
    print("Building vector store...")
    store = build_store(DOCUMENTS)
    print(f"Indexed {len(store)} chunks\n")

    # Test queries
    queries = [
        {
            "q": "What should I do when the orders pipeline fails?",
            "filter": {"type": "runbook"},
            "type": "simple"
        },
        {
            "q": "How do I optimise my Snowflake costs AND ensure data quality for the orders model?",
            "type": "complex"
        },
        {
            "q": "Who handles production incidents and what is the SLA for responding?",
            "type": "hyde"
        },
    ]

    for query_info in queries:
        q = query_info["q"]
        print(f"\n{'='*60}")
        print(f"Query: {q}")

        if query_info["type"] == "simple":
            # Compare basic vs hybrid + reranking
            print("\n[Basic semantic search]")
            q_emb = _embed([q])[0]
            basic = sorted(store, key=lambda c: c.cosine_sim(q_emb), reverse=True)[:2]
            for c in basic:
                print(f"  [{c.source}] {c.text[:80]}...")

            print("\n[Hybrid + Metadata filter]")
            filtered = technique_5_metadata_filtering(store, q, query_info.get("filter", {}))
            for c in filtered:
                print(f"  [{c.source}] {c.text[:80]}...")

        elif query_info["type"] == "complex":
            print("\n[Query decomposition]")
            answer = technique_3_query_decomposition(store, q)
            print(f"Answer: {answer[:300]}")

        elif query_info["type"] == "hyde":
            print("\n[HyDE retrieval]")
            hyde_chunks = technique_4_hyde(store, q)
            for c in hyde_chunks:
                print(f"  [{c.source}] {c.text[:80]}...")

            # Generate final answer from HyDE results
            context = "\n".join([c.text for c in hyde_chunks])
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                system="Answer using only the context. Be concise.",
                messages=[{"role": "user", "content": f"<context>{context}</context>\n\n{q}"}]
            )
            print(f"\nAnswer: {response.content[0].text.strip()[:200]}")


if __name__ == "__main__":
    main()
