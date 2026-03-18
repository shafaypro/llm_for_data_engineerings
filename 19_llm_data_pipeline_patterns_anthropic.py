"""
19_llm_data_pipeline_patterns_anthropic.py
============================================
Concept: LLM-Powered Data Pipeline Patterns
---------------------------------------------
Real production patterns for integrating LLMs into data pipelines.
These are the patterns that actually make it to production — not toys.

Patterns:
1. Schema inference from raw data
2. Automatic data quality issue triage
3. NL to SQL with execution and validation
4. Data lineage documentation generator
5. Anomaly explanation (LLM explains statistical anomalies)
6. ETL pipeline self-healing (detect + suggest fix automatically)

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json
from typing import Optional

client = anthropic.Anthropic()


# ── Pattern 1: Schema inference from raw data ─────────────────────────────────

def schema_inference(raw_records: list, table_name: str) -> dict:
    """
    Given raw JSON records, infer the Snowflake schema.
    Useful when ingesting from APIs or S3 with no schema registry.
    """
    sample = json.dumps(raw_records[:5], indent=2)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0,
        system="""You are a data engineer expert in Snowflake DDL.
Given JSON records, infer the best Snowflake schema.
Return ONLY valid JSON:
{
  "create_table_sql": "CREATE OR REPLACE TABLE ... (...)",
  "column_notes": {"col_name": "why this type was chosen"},
  "potential_issues": ["list of data quality concerns"]
}""",
        messages=[{"role": "user", "content": f"Table name: {table_name}\n\nSample records:\n{sample}"}]
    )

    return json.loads(response.content[0].text.strip())


def demo_schema_inference():
    print("=== Pattern 1: Schema Inference ===\n")

    raw_records = [
        {"order_id": "ORD-001", "customer_id": 12345, "amount": "99.99", "created_at": "2025-03-17T10:23:41Z", "tags": ["new_customer", "promo"]},
        {"order_id": "ORD-002", "customer_id": 67890, "amount": "250.00", "created_at": "2025-03-17T11:05:12Z", "tags": []},
        {"order_id": "ORD-003", "customer_id": None, "amount": "15.50", "created_at": "2025-03-17T11:45:00Z", "tags": ["promo"]},
    ]

    result = schema_inference(raw_records, "raw.stripe_orders")
    print("Inferred DDL:")
    print(result["create_table_sql"])
    print(f"\nColumn notes: {result['column_notes']}")
    print(f"Potential issues: {result['potential_issues']}")


# ── Pattern 2: Data quality issue triage ─────────────────────────────────────

def triage_dq_issues(issues: list, table_name: str, row_count: int) -> list:
    """
    Given raw dbt test failures or data quality check results,
    triage them: severity, business impact, suggested fix, owner.
    """
    issues_str = json.dumps(issues, indent=2)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=768,
        temperature=0,
        system="""You are a data quality engineer. Triage data issues.
Return a JSON array — one object per issue:
[{
  "issue": "original issue text",
  "severity": "P1|P2|P3",
  "business_impact": "one sentence on business impact",
  "root_cause_hypothesis": "most likely cause",
  "suggested_fix": "specific action",
  "fix_complexity": "quick_win|medium|major_refactor"
}]
Return ONLY the JSON array.""",
        messages=[{"role": "user", "content": f"Table: {table_name} ({row_count:,} rows)\n\nIssues:\n{issues_str}"}]
    )

    return json.loads(response.content[0].text.strip())


def demo_dq_triage():
    print("\n=== Pattern 2: Data Quality Triage ===\n")

    issues = [
        "not_null test failed on column customer_id: 8,234 null values (12.3% of rows)",
        "unique test failed on order_id: 45 duplicate values found",
        "accepted_values test failed on status: found values ['PENDING_REVIEW', 'HOLD'] not in accepted list",
        "relationship test failed: 234 order_ids in fct_orders not found in dim_customers",
    ]

    triaged = triage_dq_issues(issues, "fct_orders", 67_000)
    for item in triaged:
        print(f"[{item['severity']}] {item['issue'][:60]}...")
        print(f"  Impact: {item['business_impact']}")
        print(f"  Fix ({item['fix_complexity']}): {item['suggested_fix']}")
        print()


# ── Pattern 3: NL to SQL with execution validation ────────────────────────────

def nl_to_sql(question: str, schema_context: str, execute: bool = False) -> dict:
    """
    Convert natural language to SQL, validate it, optionally execute.
    Includes safety checks so it won't run destructive queries.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        temperature=0,
        system=f"""You are a Snowflake SQL expert. Convert questions to SQL.
RULES: SELECT only. Always include LIMIT. Use fully qualified names.
Schema:
{schema_context}

Return JSON:
{{
  "sql": "the SQL query",
  "explanation": "what this query does in plain English",
  "assumptions": ["any assumptions made"],
  "warnings": ["potential issues or edge cases"]
}}""",
        messages=[{"role": "user", "content": question}]
    )

    result = json.loads(response.content[0].text.strip())

    # Safety validation
    sql_upper = result["sql"].upper()
    if any(kw in sql_upper for kw in ["DELETE", "DROP", "TRUNCATE", "INSERT", "UPDATE"]):
        result["blocked"] = True
        result["block_reason"] = "Destructive operation detected"
        return result

    result["blocked"] = False

    if execute:
        # In production: run against your actual Snowflake connection
        result["execution_note"] = "Would execute against Snowflake (mock in demo)"

    return result


def demo_nl_to_sql():
    print("=== Pattern 3: NL to SQL ===\n")

    schema = """
    PROD_DW.MARTS.FCT_ORDERS: order_id, customer_id, order_date, total_amount, status, region
    PROD_DW.MARTS.DIM_CUSTOMERS: customer_id, name, segment, region, signup_date, lifetime_value
    """

    questions = [
        "What are the top 10 enterprise customers by revenue this month?",
        "How many orders were placed in each region last week?",
        "Delete all orders from 2020",  # should be blocked
    ]

    for q in questions:
        print(f"Q: {q}")
        result = nl_to_sql(q, schema)
        if result.get("blocked"):
            print(f"  ✗ BLOCKED: {result['block_reason']}")
        else:
            print(f"  SQL: {result['sql'][:120]}...")
            print(f"  Explanation: {result['explanation']}")
            if result.get("warnings"):
                print(f"  Warnings: {result['warnings']}")
        print()


# ── Pattern 4: Lineage documentation generator ────────────────────────────────

def generate_lineage_docs(dbt_manifest_excerpt: dict) -> str:
    """
    Given a dbt manifest (or excerpt), generate human-readable lineage docs.
    Useful for onboarding, audits, and stakeholder comms.
    """
    manifest_str = json.dumps(dbt_manifest_excerpt, indent=2)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0.3,
        system="You are a data documentation specialist. Write clear, concise lineage docs for business and engineering audiences.",
        messages=[{"role": "user", "content": f"""Write a lineage summary for these dbt models.
Format: one paragraph per model explaining what it does, where data comes from, and who uses it.

<manifest>
{manifest_str}
</manifest>"""}]
    )

    return response.content[0].text.strip()


def demo_lineage_docs():
    print("=== Pattern 4: Lineage Documentation ===\n")

    manifest = {
        "fct_orders": {
            "description": "Core fact table, one row per order",
            "depends_on": ["stg_orders", "dim_customers", "dim_products"],
            "columns": ["order_id", "customer_id", "order_date", "total_amount", "status"],
            "tags": ["daily", "critical"],
            "used_by": ["revenue_dashboard", "finance_report", "customer_success_alerts"]
        },
        "stg_orders": {
            "description": "Cleaned raw orders from MySQL",
            "depends_on": ["raw.mysql_orders"],
            "columns": ["order_id", "customer_id", "created_at", "amount_cents"],
            "tags": ["staging"]
        }
    }

    docs = generate_lineage_docs(manifest)
    print(docs)


# ── Pattern 5: Anomaly explanation ───────────────────────────────────────────

def explain_anomaly(metric_name: str, current_value: float, baseline: float,
                    context: dict) -> dict:
    """
    Given a statistical anomaly, generate a human-readable explanation
    with hypotheses ranked by likelihood. Pairs well with your existing
    anomaly detection (Great Expectations, elementary-data, etc.)
    """
    pct_change = ((current_value - baseline) / baseline) * 100

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        temperature=0.3,
        system="""You are a data analytics expert. Explain metric anomalies.
Return JSON:
{
  "summary": "one sentence for standup",
  "hypotheses": [{"hypothesis": "text", "likelihood": "high|medium|low", "how_to_verify": "action"}],
  "immediate_actions": ["action1", "action2"],
  "escalate": true/false
}""",
        messages=[{"role": "user", "content": f"""Metric: {metric_name}
Current: {current_value:,.0f}
Baseline (7-day avg): {baseline:,.0f}
Change: {pct_change:+.1f}%

Context:
{json.dumps(context, indent=2)}"""}]
    )

    return json.loads(response.content[0].text.strip())


def demo_anomaly_explanation():
    print("\n=== Pattern 5: Anomaly Explanation ===\n")

    result = explain_anomaly(
        metric_name="daily_order_count",
        current_value=312,
        baseline=1_247,
        context={
            "day_of_week": "Monday",
            "is_holiday": False,
            "recent_deployments": ["promo_code_feature deployed 6pm yesterday"],
            "pipeline_health": "all DAGs green",
            "payment_processor_status": "no incidents reported",
            "marketing_campaigns": "no active campaigns"
        }
    )

    print(f"Summary: {result['summary']}")
    print(f"\nHypotheses:")
    for h in result["hypotheses"]:
        print(f"  [{h['likelihood'].upper()}] {h['hypothesis']}")
        print(f"    Verify: {h['how_to_verify']}")
    print(f"\nImmediate actions: {result['immediate_actions']}")
    print(f"Escalate: {result['escalate']}")


# ── Pattern 6: Self-healing pipeline suggestions ──────────────────────────────

def suggest_pipeline_fix(dag_id: str, task_id: str, error_log: str,
                          historical_fixes: list = None) -> dict:
    """
    Given a pipeline failure + optional history of past fixes,
    suggest a specific fix and an Airflow config change if applicable.
    """
    context = ""
    if historical_fixes:
        context = f"\nHistorical fixes for similar errors:\n" + "\n".join(historical_fixes)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        temperature=0,
        system=f"""You are an Airflow/data pipeline reliability engineer.
Analyse failures and suggest specific fixes.{context}
Return JSON:
{{
  "root_cause": "technical explanation",
  "fix_steps": ["step 1", "step 2"],
  "airflow_config_change": {{"setting": "new_value"}} or null,
  "retry_safe": true/false,
  "prevention": "how to prevent recurrence",
  "estimated_fix_time": "5min|30min|2hr|1day"
}}""",
        messages=[{"role": "user", "content": f"DAG: {dag_id} | Task: {task_id}\n\nError:\n{error_log}"}]
    )

    return json.loads(response.content[0].text.strip())


def demo_self_healing():
    print("\n=== Pattern 6: Self-Healing Pipeline Suggestions ===\n")

    error_log = """
    OperationalError: (snowflake.connector.errors.OperationalError) 250001 (08001):
    Failed to connect to DB: account=company.us-east-1, user=svc_airflow
    SSL SYSCALL error: Connection reset by peer
    Connection timeout after 30s. Retried 3 times.
    """

    historical_fixes = [
        "2025-02-14: Same error — fixed by increasing connection timeout to 60s in airflow.cfg",
        "2025-01-08: Connection reset — root cause was Snowflake warehouse cold start, added 10s retry delay",
    ]

    fix = suggest_pipeline_fix(
        dag_id="daily_orders_etl",
        task_id="load_to_snowflake",
        error_log=error_log,
        historical_fixes=historical_fixes
    )

    print(f"Root cause: {fix['root_cause']}")
    print(f"\nFix steps:")
    for i, step in enumerate(fix["fix_steps"], 1):
        print(f"  {i}. {step}")
    if fix.get("airflow_config_change"):
        print(f"\nConfig change: {fix['airflow_config_change']}")
    print(f"\nRetry safe: {fix['retry_safe']}")
    print(f"Estimated fix time: {fix['estimated_fix_time']}")
    print(f"Prevention: {fix['prevention']}")


if __name__ == "__main__":
    demo_schema_inference()
    demo_dq_triage()
    demo_nl_to_sql()
    demo_lineage_docs()
    demo_anomaly_explanation()
    demo_self_healing()
