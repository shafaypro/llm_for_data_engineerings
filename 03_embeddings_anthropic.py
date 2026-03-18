"""
03_embeddings_anthropic.py
==========================
Concept: Embeddings
-------------------
Embeddings convert text into a fixed-length array of floats that encodes
semantic meaning. Similar texts have similar vectors (small cosine distance).
This is the foundation of RAG — you embed your docs, store the vectors,
then at query time find the nearest ones to inject into context.

Think of it like a feature vector in your ML work — except the model
learns the representation, not you.

Key facts:
- Voyage (Anthropic's embedding partner) produces 1024-dim vectors
- Cosine similarity measures how "close" two texts are semantically
- "king" - "man" + "woman" ≈ "queen" (the classic demo)
- Stored in vector DBs: pgvector, Pinecone, Weaviate, Chroma

Install: pip install anthropic voyageai numpy
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
          export VOYAGE_API_KEY=pa-...   (get from voyageai.com)
"""

import numpy as np

# NOTE: Voyage AI is Anthropic's recommended embedding partner.
# pip install voyageai
try:
    import voyageai
    vo = voyageai.Client()
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False
    print("voyageai not installed — running with mock embeddings for demo.")
    print("Install with: pip install voyageai\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """Get a real embedding via Voyage, or a mock one for demo purposes."""
    if VOYAGE_AVAILABLE:
        result = vo.embed([text], model="voyage-3-lite")
        return np.array(result.embeddings[0])
    else:
        # Mock: deterministic fake embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(1024)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Range: -1 to 1, higher = more similar."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Demo 1: Semantic similarity ────────────────────────────────────────────────
def semantic_similarity_demo():
    """
    Show that semantically similar texts have high cosine similarity
    even when they use different words.
    """

    texts = [
        "The Airflow DAG failed due to a database connection timeout",
        "Pipeline task crashed because it could not reach the database",  # semantically similar
        "The quarterly revenue report is ready for review",               # unrelated
        "SELECT * FROM orders WHERE status = 'failed'",                   # SQL, somewhat related
    ]

    print("=== Semantic Similarity Demo ===")
    print(f"Base text: '{texts[0]}'\n")

    base_emb = get_embedding(texts[0])
    for text in texts[1:]:
        emb = get_embedding(text)
        sim = cosine_similarity(base_emb, emb)
        bar = "█" * int(sim * 30)
        print(f"  Similarity {sim:.3f} {bar}")
        print(f"  '{text[:60]}...'\n" if len(text) > 60 else f"  '{text}'\n")


# ── Demo 2: Build a tiny in-memory vector store ────────────────────────────────
def vector_store_demo():
    """
    A minimal vector store: embed documents, store vectors,
    search by cosine similarity. This is what pgvector / Pinecone do
    under the hood — just at much larger scale.
    """

    # Your "knowledge base" — could be runbooks, dbt docs, schema docs
    documents = [
        {"id": "doc1", "text": "The orders DAG runs at 2am UTC and loads from the Postgres replica into Snowflake."},
        {"id": "doc2", "text": "If the orders DAG fails, check the RDS replica lag first. Common cause is replication delay."},
        {"id": "doc3", "text": "The revenue model in dbt depends on fct_orders and dim_customers."},
        {"id": "doc4", "text": "Snowflake warehouse sizing: use SMALL for dbt runs under 10 mins, MEDIUM for full refreshes."},
        {"id": "doc5", "text": "On-call rotation: week 1 = Alice, week 2 = Bob, week 3 = Carlos."},
    ]

    print("=== Vector Store Demo ===")
    print("Embedding documents...")
    for doc in documents:
        doc["embedding"] = get_embedding(doc["text"])
    print(f"Embedded {len(documents)} documents.\n")

    def search(query: str, top_k: int = 2) -> list:
        q_emb = get_embedding(query)
        scored = [(cosine_similarity(q_emb, doc["embedding"]), doc) for doc in documents]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    queries = [
        "What should I do when the pipeline fails?",
        "How big should my Snowflake warehouse be?",
        "Who is on call this week?",
    ]

    for query in queries:
        print(f"Query: '{query}'")
        results = search(query, top_k=2)
        for score, doc in results:
            print(f"  [{score:.3f}] {doc['text'][:80]}")
        print()


# ── Demo 3: Embed then answer with Claude (mini RAG) ──────────────────────────
def embed_then_answer_demo():
    """
    Combine embeddings + Claude: retrieve relevant docs, then answer.
    This is the core pattern of RAG.
    """
    import anthropic
    client = anthropic.Anthropic()

    docs = [
        "The fct_orders dbt model runs in the marts layer and has a 2-hour SLA.",
        "Database connection errors in Airflow are usually fixed by restarting the task with exponential backoff.",
        "Our Snowflake account is in us-east-1. Queries over 60 seconds auto-cancel.",
        "The orders pipeline loads data from MySQL via Fivetran, landing in raw.fivetran_orders.",
    ]

    print("=== Embed-then-Answer Demo ===")
    embeddings = [get_embedding(d) for d in docs]

    def rag_answer(question: str) -> str:
        q_emb = get_embedding(question)
        scores = [cosine_similarity(q_emb, e) for e in embeddings]
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
        context = "\n".join([docs[i] for i in top_idx])

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system="Answer using only the provided context. Be concise.",
            messages=[{"role": "user", "content": f"<context>\n{context}\n</context>\n\n{question}"}]
        )
        return resp.content[0].text

    questions = [
        "Where does the orders data originally come from?",
        "What's the SLA for the orders model?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {rag_answer(q)}\n")


if __name__ == "__main__":
    semantic_similarity_demo()
    vector_store_demo()
    embed_then_answer_demo()
