"""
01_context_window_anthropic.py
==============================
Concept: Context Window
-----------------------
The context window is everything the model can "see" in one API call —
your system prompt, conversation history, documents, and the user question.
Think of it like the working memory of a single SQL query execution.

Key facts:
- Claude Sonnet: 200,000 tokens (~150,000 words)
- 1 token ≈ 0.75 words ≈ 4 characters
- If you exceed it, the call fails or older content gets dropped
- You pay per token (input + output)

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic

client = anthropic.Anthropic()


# ── Demo 1: Basic context window usage ────────────────────────────────────────
def basic_context_demo():
    """Show how system prompt + user message fill the context window."""

    system_prompt = """You are a senior data engineer at a fintech company.
Our stack: Airflow, dbt, Snowflake, AWS, Python.
Always give concrete, runnable examples."""

    user_message = "What's the best way to handle late-arriving data in my pipelines?"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system_prompt,          # <-- fills part of the context window
        messages=[{
            "role": "user",
            "content": user_message    # <-- fills more of the context window
        }]
    )

    print("=== Basic Context Demo ===")
    print(response.content[0].text)
    print(f"\nTokens used — input: {response.usage.input_tokens}, output: {response.usage.output_tokens}")


# ── Demo 2: Large document in context ─────────────────────────────────────────
def large_document_demo():
    """
    Pass a large document (e.g. a dbt model SQL file) into the context window.
    The model can answer questions about it without any vector DB or RAG.
    Only works because Claude has a 200k token window.
    """

    # Simulate a large dbt model SQL file
    large_sql_document = """
    -- models/marts/finance/fct_revenue.sql
    -- This model calculates daily revenue metrics broken down by product, region, and customer segment.

    WITH orders AS (
        SELECT * FROM {{ ref('stg_orders') }}
        WHERE status = 'completed'
    ),
    order_items AS (
        SELECT * FROM {{ ref('stg_order_items') }}
    ),
    products AS (
        SELECT * FROM {{ ref('dim_products') }}
    ),
    customers AS (
        SELECT * FROM {{ ref('dim_customers') }}
    ),
    joined AS (
        SELECT
            o.order_id,
            o.created_at::DATE AS order_date,
            o.customer_id,
            c.region,
            c.segment AS customer_segment,
            oi.product_id,
            p.category AS product_category,
            p.subcategory AS product_subcategory,
            oi.quantity,
            oi.unit_price,
            oi.discount_pct,
            oi.quantity * oi.unit_price * (1 - oi.discount_pct) AS net_revenue,
            oi.quantity * oi.unit_price AS gross_revenue
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        JOIN customers c ON o.customer_id = c.customer_id
    )
    SELECT
        order_date,
        region,
        customer_segment,
        product_category,
        product_subcategory,
        COUNT(DISTINCT order_id) AS order_count,
        COUNT(DISTINCT customer_id) AS unique_customers,
        SUM(quantity) AS total_units_sold,
        SUM(gross_revenue) AS total_gross_revenue,
        SUM(net_revenue) AS total_net_revenue,
        SUM(gross_revenue - net_revenue) AS total_discount_amount,
        AVG(net_revenue / NULLIF(quantity, 0)) AS avg_net_revenue_per_unit
    FROM joined
    GROUP BY 1, 2, 3, 4, 5
    """ * 3  # repeat to simulate a large file

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""Here is a dbt SQL model. Answer these questions:
1. What grain is this model at?
2. What are the key metrics it calculates?
3. What upstream models does it depend on?

<sql_model>
{large_sql_document}
</sql_model>"""
        }]
    )

    print("\n=== Large Document in Context Demo ===")
    print(response.content[0].text)
    print(f"\nTokens used — input: {response.usage.input_tokens}")


# ── Demo 3: Context window management — truncation strategy ───────────────────
def context_management_demo():
    """
    Show how to handle content that might exceed the context window.
    Strategy: keep the most recent / most relevant content, truncate the rest.
    """

    def truncate_to_token_budget(text: str, max_chars: int = 50000) -> str:
        """
        Simple truncation strategy: keep the LAST max_chars characters.
        For logs and error messages, errors are usually at the end.
        For documents, you'd want a smarter chunking strategy (see RAG example).
        """
        if len(text) <= max_chars:
            return text
        truncated = text[-max_chars:]
        return f"[... truncated {len(text) - max_chars} characters ...]\n\n{truncated}"

    long_log = "INFO: task started\n" * 5000 + "ERROR: connection timeout after 30s\nKeyError: 'order_id' not found"

    safe_log = truncate_to_token_budget(long_log, max_chars=2000)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"What error occurred in this log?\n\n<log>{safe_log}</log>"
        }]
    )

    print("\n=== Context Management Demo ===")
    print(f"Original log length: {len(long_log):,} chars")
    print(f"Truncated to: {len(safe_log):,} chars")
    print(f"Answer: {response.content[0].text}")


if __name__ == "__main__":
    basic_context_demo()
    large_document_demo()
    context_management_demo()
