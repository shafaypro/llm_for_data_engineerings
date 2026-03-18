"""
06_rag_anthropic.py
===================
Concept: RAG — Retrieval Augmented Generation
----------------------------------------------
RAG solves the problem that LLMs have a training cutoff and know nothing
about your internal data. Instead of retraining, you:
  1. Chunk your documents
  2. Embed each chunk (convert to a vector)
  3. Store vectors in a vector DB
  4. At query time: embed the question, find the nearest chunks
  5. Inject those chunks into the context window, then call the LLM

Think of it like a parameterised SQL query that first fetches the relevant
rows, then passes them to a function. The model is the function.

Install: pip install anthropic voyageai numpy
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
          export VOYAGE_API_KEY=pa-...
"""

import anthropic
import numpy as np
import json
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
        # Mock embeddings for demo
        return [np.random.RandomState(hash(t) % 2**32).randn(512) for t in texts]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Document:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


# ── Step 1: Chunking ───────────────────────────────────────────────────────────

def chunk_document(text: str, source: str, chunk_size: int = 300, overlap: int = 50) -> List[Document]:
    """
    Split a document into overlapping chunks.
    overlap ensures context isn't lost at chunk boundaries —
    same idea as a sliding window in stream processing.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i:i + chunk_size])
        if len(chunk_text.strip()) < 50:  # skip tiny trailing chunks
            continue
        chunks.append(Document(
            text=chunk_text,
            source=source,
            metadata={"chunk_index": len(chunks), "word_start": i}
        ))
    return chunks


# ── Step 2 & 3: Embed and store ───────────────────────────────────────────────

class VectorStore:
    """
    In-memory vector store. In production, replace with:
    - pgvector (if you're already on Postgres/Snowflake)
    - Pinecone (managed, scales well)
    - Chroma (local dev, easy setup)
    - Weaviate (open source, self-hosted)
    """

    def __init__(self):
        self.docs: List[Document] = []

    def add_documents(self, docs: List[Document]) -> None:
        """Embed all documents and store."""
        texts = [d.text for d in docs]
        embeddings = _embed(texts)
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb
        self.docs.extend(docs)
        print(f"  Added {len(docs)} chunks. Total: {len(self.docs)}")

    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.0) -> List[tuple]:
        """Find top_k most similar documents by cosine similarity."""
        q_emb = _embed([query])[0]
        scored = []
        for doc in self.docs:
            sim = float(np.dot(q_emb, doc.embedding) /
                        (np.linalg.norm(q_emb) * np.linalg.norm(doc.embedding) + 1e-9))
            if sim >= min_similarity:
                scored.append((sim, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# ── Step 4 & 5: Retrieve and generate ────────────────────────────────────────

def rag_query(store: VectorStore, question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Build context string
    3. Call Claude with context + question
    4. Return answer + sources
    """

    # Step 4: Retrieve
    results = store.search(question, top_k=top_k)

    if not results:
        return {"answer": "No relevant information found.", "sources": []}

    # Build context block — XML tags help Claude understand the structure
    context_parts = []
    for i, (score, doc) in enumerate(results, 1):
        context_parts.append(f"<source_{i} file='{doc.source}' relevance='{score:.2f}'>\n{doc.text}\n</source_{i}>")
    context = "\n\n".join(context_parts)

    # Step 5: Generate
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="""You are a data engineering assistant with access to internal documentation.
Answer questions using ONLY the provided source documents.
If the answer isn't in the sources, say "I don't have that information in the provided docs."
Always cite which source you used.""",
        messages=[{
            "role": "user",
            "content": f"<documentation>\n{context}\n</documentation>\n\nQuestion: {question}"
        }]
    )

    return {
        "answer": response.content[0].text,
        "sources": [(score, doc.source) for score, doc in results],
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens
    }


# ── Full demo ─────────────────────────────────────────────────────────────────

def main():
    print("=== RAG Pipeline Demo ===\n")

    # Simulate your internal documentation
    runbook = """
    Airflow Pipeline Operations Runbook

    DAG: daily_orders_etl
    Schedule: 2:00 AM UTC daily
    SLA: Must complete by 4:00 AM UTC
    Owner: data-engineering@company.com

    What it does:
    This DAG extracts order data from the MySQL production replica, transforms it,
    and loads it into the Snowflake data warehouse ORDERS.MARTS.FCT_ORDERS table.

    Common failures and fixes:
    1. MySQL connection timeout: Usually caused by replica lag. Check replica lag in
       CloudWatch. If lag > 60 seconds, wait 10 minutes and manually trigger.
    2. Snowflake credit exhaustion: Check SNOWFLAKE_USAGE dashboard. Escalate to
       platform team if credits < 100.
    3. Schema mismatch error: Run 'dbt run --select stg_orders' to refresh the staging
       model, then re-trigger the DAG.
    4. Duplicate key violation: Run the dedup script at scripts/dedup_orders.sql
       then re-trigger from the failed task.

    Escalation path: data-engineering Slack channel → on-call engineer → CTO
    """

    dbt_docs = """
    dbt Model Documentation

    Model: fct_orders
    Layer: marts
    Grain: one row per order_id
    SLA: refreshed daily by 4 AM UTC
    Depends on: stg_orders, dim_customers, dim_products

    Columns:
    - order_id: unique identifier for the order (UUID)
    - customer_id: foreign key to dim_customers
    - order_date: date the order was placed (UTC, date grain)
    - total_amount: gross order value before discounts, in USD
    - net_amount: order value after discounts applied, in USD
    - status: current order status (pending/completed/cancelled/refunded)
    - region: customer region from dim_customers (APAC/EMEA/AMER)

    Known issues:
    - Orders from before 2022-01-01 may have null product_id due to legacy system migration
    - Refunded orders keep their original total_amount; use status='refunded' to filter

    Model: dim_customers
    Layer: marts
    Grain: one row per customer_id (latest snapshot)
    Columns:
    - customer_id, name, email, region, segment, signup_date, is_active, lifetime_value
    """

    schema_docs = """
    Data Warehouse Schema Reference

    Database: PROD_DW
    Warehouse: TRANSFORM_WH (use SMALL for reads, MEDIUM for full refreshes)

    Key schemas:
    - RAW: raw data as-landed from Fivetran connectors. Never query directly.
    - STAGING: cleaned, typed, deduplicated source data. stg_ prefix.
    - MARTS: business-ready models. fct_ for facts, dim_ for dimensions.

    Access:
    - Analysts: read access to MARTS only
    - Data engineers: full access to all schemas
    - BI tools: connect via service account DE_BI_USER, MARTS schema only

    Cost controls:
    - Auto-suspend: 5 minutes idle
    - Statement timeout: 10 minutes (override in session if needed)
    - Max cluster count: 3 (contact platform team to raise)

    Monitoring: Snowflake dashboards at snowflake.company.com
    Alerts: #data-alerts Slack channel
    """

    # Build the vector store
    store = VectorStore()
    print("Indexing documents...")
    for text, source in [(runbook, "ops_runbook.md"), (dbt_docs, "dbt_docs.md"), (schema_docs, "schema_ref.md")]:
        chunks = chunk_document(text, source, chunk_size=150, overlap=30)
        store.add_documents(chunks)

    print(f"\nIndexed {len(store.docs)} total chunks\n")
    print("-" * 60)

    # Query the knowledge base
    questions = [
        "What should I do when the orders DAG fails with a MySQL timeout?",
        "What is the grain of the fct_orders model?",
        "Which Snowflake warehouse size should I use for dbt runs?",
        "How do I access the data warehouse as an analyst?",
        "What was the company's revenue last year?",  # not in docs — should say so
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = rag_query(store, question)
        print(f"A: {result['answer'][:400].strip()}")
        print(f"Sources: {[s for _, s in result['sources']]}")
        print(f"Tokens: {result['tokens_used']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
