"""
10_cost_and_caching_anthropic.py
=================================
Concept: Cost, Token Pricing & Prompt Caching
----------------------------------------------
API calls cost money per token. For a pipeline running 10,000 times/day
with a large system prompt, unoptimised token usage adds up fast.

Key levers:
1. Use the smallest model that works for the task
2. Keep prompts tight — every token costs money
3. Use prompt caching for repeated system prompts (~90% cheaper on cached tokens)
4. Use the Batch API for async workloads (~50% cheaper)
5. Set max_tokens tightly for structured outputs

Pricing ballpark (check docs.anthropic.com for current rates):
- Haiku:  $0.25/M input,  $1.25/M output
- Sonnet: $3.00/M input,  $15.00/M output
- Opus:   $15.00/M input, $75.00/M output
- Prompt cache write: 25% more than base input price
- Prompt cache read: 10% of base input price (90% saving!)

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import time
import json

client = anthropic.Anthropic()

# Approximate pricing per token (update from docs.anthropic.com)
PRICING = {
    "claude-haiku-4-5-20251001":  {"input": 0.25e-6,  "output": 1.25e-6},
    "claude-sonnet-4-6": {"input": 3.00e-6,  "output": 15.00e-6},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """Estimate API call cost in USD."""
    p = PRICING.get(model, PRICING["claude-haiku-4-5-20251001"])
    cost = (input_tokens * p["input"] +
            output_tokens * p["output"] +
            cache_read_tokens * p["input"] * 0.1 +       # 90% cheaper
            cache_write_tokens * p["input"] * 1.25)       # 25% more to write
    return cost


# ── Demo 1: Model selection impact on cost ────────────────────────────────────
def model_selection_demo():
    """
    Compare cost of the same task across model tiers.
    For most data engineering tasks, Haiku is good enough.
    """

    print("=== Model Selection Cost Comparison ===\n")

    task = "Classify this as P1/P2/P3 and return JSON: 'Snowflake warehouse auto-suspended'"
    system = 'Respond ONLY with JSON: {"severity": "P1|P2|P3", "reason": "one sentence"}'

    for model in ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]:
        response = client.messages.create(
            model=model,
            max_tokens=64,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": task}]
        )
        cost = estimate_cost(model, response.usage.input_tokens, response.usage.output_tokens)
        print(f"Model: {model}")
        print(f"  Output: {response.content[0].text.strip()}")
        print(f"  Tokens: {response.usage.input_tokens} in + {response.usage.output_tokens} out")
        print(f"  Cost: ${cost:.8f} per call")
        print(f"  At 10k calls/day: ${cost * 10000:.4f}/day\n")


# ── Demo 2: Prompt caching — massive savings for repeated system prompts ───────
def prompt_caching_demo():
    """
    Prompt caching is the highest-impact cost optimisation for pipelines.
    If your system prompt contains a large schema or runbook, Claude caches
    it after the first call and charges only 10% on subsequent reads.

    To enable: add {"cache_control": {"type": "ephemeral"}} to the last
    content block you want cached.
    """

    print("=== Prompt Caching Demo ===\n")

    # Simulate a large system prompt with your full schema definition
    # In production this might be 5,000-20,000 tokens of schema docs
    large_schema = """
    You are a SQL assistant for our Snowflake data warehouse. Use only these tables:

    PROD_DW.MARTS.FCT_ORDERS:
      - order_id UUID NOT NULL PRIMARY KEY
      - customer_id UUID NOT NULL REFERENCES dim_customers(customer_id)
      - order_date DATE NOT NULL
      - total_amount DECIMAL(18,2) NOT NULL
      - net_amount DECIMAL(18,2) NOT NULL
      - status VARCHAR(20) -- values: pending, completed, cancelled, refunded
      - region VARCHAR(10) -- values: AMER, EMEA, APAC
      - created_at TIMESTAMP_NTZ NOT NULL
      - updated_at TIMESTAMP_NTZ NOT NULL

    PROD_DW.MARTS.DIM_CUSTOMERS:
      - customer_id UUID NOT NULL PRIMARY KEY
      - name VARCHAR(255) NOT NULL
      - email VARCHAR(255) NOT NULL UNIQUE
      - region VARCHAR(10) NOT NULL
      - segment VARCHAR(50) -- values: enterprise, mid-market, smb
      - signup_date DATE NOT NULL
      - lifetime_value DECIMAL(18,2)
      - is_active BOOLEAN NOT NULL DEFAULT TRUE

    PROD_DW.MARTS.FCT_REVENUE:
      - revenue_date DATE NOT NULL
      - region VARCHAR(10) NOT NULL
      - product_category VARCHAR(100) NOT NULL
      - gross_revenue DECIMAL(18,2) NOT NULL
      - net_revenue DECIMAL(18,2) NOT NULL
      - order_count INTEGER NOT NULL
      - unique_customers INTEGER NOT NULL

    Always use fully qualified table names. Only write SELECT statements. Always include LIMIT.
    """ * 5  # multiply to make it realistically large

    questions = [
        "Show top 5 customers by revenue this month",
        "What's the daily revenue trend for the last 7 days?",
        "How many orders are in 'pending' status?",
    ]

    print(f"System prompt size: ~{len(large_schema.split())} words\n")

    total_no_cache = 0
    total_with_cache = 0

    for i, question in enumerate(questions):
        # Without caching
        resp_no_cache = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0,
            system=large_schema,
            messages=[{"role": "user", "content": question}]
        )
        cost_no_cache = estimate_cost(
            "claude-haiku-4-5-20251001",
            resp_no_cache.usage.input_tokens,
            resp_no_cache.usage.output_tokens
        )
        total_no_cache += cost_no_cache

        # With prompt caching
        resp_cached = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0,
            system=[{
                "type": "text",
                "text": large_schema,
                "cache_control": {"type": "ephemeral"}  # <-- this is all you need
            }],
            messages=[{"role": "user", "content": question}]
        )

        cache_read = getattr(resp_cached.usage, "cache_read_input_tokens", 0)
        cache_write = getattr(resp_cached.usage, "cache_creation_input_tokens", 0)
        cost_cached = estimate_cost(
            "claude-haiku-4-5-20251001",
            resp_cached.usage.input_tokens,
            resp_cached.usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write
        )
        total_with_cache += cost_cached

        print(f"Q{i+1}: {question[:50]}...")
        print(f"  No cache: ${cost_no_cache:.8f} ({resp_no_cache.usage.input_tokens} input tokens)")
        print(f"  Cached:   ${cost_cached:.8f} (cache_read={cache_read}, cache_write={cache_write})")
        print()

    print(f"Total for 3 queries:")
    print(f"  Without caching: ${total_no_cache:.6f}")
    print(f"  With caching:    ${total_with_cache:.6f}")
    if total_no_cache > 0:
        saving_pct = (1 - total_with_cache / total_no_cache) * 100
        print(f"  Saving: {saving_pct:.1f}%")
        print(f"  At 10k queries/day: ${(total_no_cache - total_with_cache) * 10000/3:.2f}/day saved")


# ── Demo 3: Batch API for async workloads ────────────────────────────────────
def batch_api_demo():
    """
    The Batch API processes requests asynchronously at ~50% of normal price.
    Perfect for: nightly dbt doc generation, bulk alert classification,
    processing thousands of records that don't need immediate results.

    Note: results are available within 24 hours, not in real-time.
    """

    print("\n=== Batch API Demo (50% cheaper for async jobs) ===\n")

    # Prepare a batch of requests — e.g. classify 5 alerts
    alerts = [
        "CRITICAL: Primary database connection pool exhausted",
        "WARNING: dbt model fct_orders exceeded SLA by 15 minutes",
        "INFO: Fivetran sync completed. 50,000 rows ingested.",
        "ERROR: AWS S3 access denied for data lake bucket",
        "WARNING: Snowflake query running for over 30 minutes",
    ]

    requests = [
        {
            "custom_id": f"alert_{i}",
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 64,
                "temperature": 0,
                "system": 'Classify alerts. Respond ONLY with JSON: {"severity": "P1|P2|P3", "category": "infra|data_quality|performance"}',
                "messages": [{"role": "user", "content": alert}]
            }
        }
        for i, alert in enumerate(alerts)
    ]

    print(f"Submitting batch of {len(requests)} requests...")
    print("(In production these would be processed overnight at 50% cost)\n")

    # Submit the batch
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"Request counts: {batch.request_counts}")

    # Poll until complete (for demo — in production use webhooks or check next morning)
    print("\nPolling for results (may take a few minutes)...")
    while batch.processing_status == "in_progress":
        time.sleep(5)
        batch = client.messages.batches.retrieve(batch.id)
        print(f"  Status: {batch.processing_status} — {batch.request_counts}")

    # Retrieve and display results
    if batch.processing_status == "ended":
        print("\nResults:")
        for result in client.messages.batches.results(batch.id):
            if result.result.type == "succeeded":
                alert_idx = int(result.custom_id.split("_")[1])
                output = json.loads(result.result.message.content[0].text)
                print(f"  Alert {alert_idx}: {alerts[alert_idx][:50]}...")
                print(f"    → {output}")


# ── Demo 4: Cost calculator ────────────────────────────────────────────────────
def cost_calculator():
    """Quick cost estimation for your pipeline before building it."""

    print("\n=== Pipeline Cost Calculator ===\n")

    scenarios = [
        {
            "name": "Airflow log classifier (background job)",
            "model": "claude-haiku-4-5-20251001",
            "calls_per_day": 500,
            "avg_input_tokens": 800,
            "avg_output_tokens": 100,
        },
        {
            "name": "dbt doc generator (nightly batch)",
            "model": "claude-haiku-4-5-20251001",
            "calls_per_day": 200,
            "avg_input_tokens": 400,
            "avg_output_tokens": 300,
        },
        {
            "name": "Interactive SQL assistant (user-facing)",
            "model": "claude-sonnet-4-6",
            "calls_per_day": 1000,
            "avg_input_tokens": 2000,
            "avg_output_tokens": 500,
        },
    ]

    print(f"{'Scenario':<45} {'Model':<12} {'$/day':>8}  {'$/month':>10}")
    print("-" * 85)

    for s in scenarios:
        p = PRICING.get(s["model"], PRICING["claude-haiku-4-5-20251001"])
        cost_per_call = s["avg_input_tokens"] * p["input"] + s["avg_output_tokens"] * p["output"]
        cost_per_day = cost_per_call * s["calls_per_day"]
        cost_per_month = cost_per_day * 30
        model_short = s["model"].split("-")[1]
        print(f"{s['name']:<45} {model_short:<12} ${cost_per_day:>7.2f}  ${cost_per_month:>9.2f}")


if __name__ == "__main__":
    model_selection_demo()
    prompt_caching_demo()
    batch_api_demo()
    cost_calculator()
