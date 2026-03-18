"""
04_temperature_anthropic.py
===========================
Concept: Temperature
--------------------
Temperature controls how deterministic vs creative the model's output is.
Under the hood it scales the probability distribution over next tokens before
sampling — low temp = peaky distribution (confident), high temp = flat (varied).

Key rules for data engineers:
- temperature=0   → deterministic, use for SQL generation, JSON extraction, classification
- temperature=0.3 → slightly varied, good for summaries with some style
- temperature=0.7 → creative but coherent, good for reports, docs
- temperature=1+  → very varied, use for brainstorming, ideation only

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json

client = anthropic.Anthropic()


# ── Demo 1: SQL generation — always use temperature=0 ─────────────────────────
def sql_generation_demo():
    """
    For structured outputs like SQL, temperature=0 is essential.
    You want the same correct query every time, not creative variations.
    """

    prompt = """Write a SQL query that finds the top 5 customers by total revenue
    in the last 30 days. Tables: orders(id, customer_id, created_at, total_amount),
    customers(id, name, email). Return only the SQL, no explanation."""

    print("=== SQL Generation: temperature=0 vs temperature=1 ===\n")

    for temp in [0, 1]:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"temperature={temp}:")
        print(response.content[0].text.strip())
        print()


# ── Demo 2: Classification — temperature=0 ────────────────────────────────────
def classification_demo():
    """
    For classification tasks (e.g. tagging pipeline alerts by severity),
    always use temperature=0 for consistent, reproducible results.
    """

    alerts = [
        "CRITICAL: Snowflake warehouse suspended due to credit exhaustion",
        "WARNING: dbt model fct_orders took 45 minutes, SLA is 30 minutes",
        "INFO: Daily ETL completed successfully. 1.2M rows loaded.",
        "ERROR: Fivetran sync failed: MySQL connection refused",
    ]

    print("=== Alert Classification: temperature=0 ===\n")

    for alert in alerts:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            temperature=0,  # deterministic classification
            system="""Classify data pipeline alerts. Respond ONLY with valid JSON:
{"severity": "P1|P2|P3", "category": "infra|data_quality|performance|auth"}""",
            messages=[{"role": "user", "content": alert}]
        )
        result = json.loads(response.content[0].text)
        print(f"Alert: {alert[:60]}...")
        print(f"  → {result}\n")


# ── Demo 3: Report writing — temperature=0.7 ──────────────────────────────────
def report_writing_demo():
    """
    For narrative writing like pipeline status reports or runbook summaries,
    a higher temperature produces more natural, varied prose.
    Run it twice to see different outputs.
    """

    pipeline_stats = {
        "dag_id": "daily_revenue_etl",
        "date": "2025-03-17",
        "rows_processed": 1_247_832,
        "duration_minutes": 23,
        "sla_minutes": 30,
        "failed_tasks": 0,
        "warnings": ["partition p_2025_03_16 has 3% null rate in customer_id column"]
    }

    prompt = f"""Write a 3-sentence pipeline health summary for the data team daily standup.
Pipeline stats: {json.dumps(pipeline_stats, indent=2)}"""

    print("=== Report Writing: temperature=0.7 (run twice) ===\n")

    for i in range(2):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"Run {i+1}:")
        print(response.content[0].text.strip())
        print()


# ── Demo 4: Brainstorming — temperature=1 ─────────────────────────────────────
def brainstorming_demo():
    """
    For ideation — e.g. "what could cause this data anomaly?" —
    a high temperature generates more diverse hypotheses.
    You wouldn't use this output directly in production, but for
    exploring failure modes it's genuinely useful.
    """

    prompt = """Give 5 possible reasons why a daily revenue metric could suddenly
    drop by 40% overnight. Be creative and consider unusual causes too."""

    print("=== Brainstorming: temperature=1 ===\n")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=1,
        messages=[{"role": "user", "content": prompt}]
    )
    print(response.content[0].text.strip())


# ── Demo 5: The right temperature by task type ────────────────────────────────
def temperature_guide():
    """Quick reference: correct temperature for common data engineering tasks."""

    tasks = [
        ("SQL generation",         0.0, "Must be correct and consistent"),
        ("JSON data extraction",   0.0, "Schema must match exactly"),
        ("Alert classification",   0.0, "Reproducible results required"),
        ("Error log summarisation",0.3, "Accurate but readable"),
        ("dbt model documentation",0.5, "Factual with some natural prose"),
        ("Pipeline status report", 0.7, "Natural writing, varied phrasing"),
        ("Root cause brainstorming",1.0, "Maximum diversity of ideas"),
    ]

    print("=== Temperature Guide for Data Engineering Tasks ===\n")
    print(f"{'Task':<35} {'Temp':>5}  {'Reason'}")
    print("-" * 80)
    for task, temp, reason in tasks:
        bar = "▓" * int(temp * 10) + "░" * (10 - int(temp * 10))
        print(f"{task:<35} {temp:>5.1f}  [{bar}] {reason}")


if __name__ == "__main__":
    sql_generation_demo()
    classification_demo()
    report_writing_demo()
    brainstorming_demo()
    temperature_guide()
