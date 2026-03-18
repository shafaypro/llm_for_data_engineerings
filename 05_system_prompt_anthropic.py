"""
05_system_prompt_anthropic.py
=============================
Concept: System Prompt
----------------------
The system prompt is your config file — it runs before every user message
and defines the model's persona, constraints, output format, and domain context.
Users don't see it, but it shapes every single response.

Think of it like an Airflow Variable or environment config: set it once,
it applies to the whole pipeline run.

Key patterns:
- Role framing: "You are a senior data engineer at..."
- Output contract: "Always respond with valid JSON in this shape: ..."
- Constraints: "Never write DELETE or DROP statements"
- Domain context: "Our stack is Snowflake + dbt + Airflow on AWS"
- Few-shot examples: show 1-2 examples of ideal input/output pairs

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json

client = anthropic.Anthropic()


# ── Demo 1: Role framing changes answer quality ────────────────────────────────
def role_framing_demo():
    """
    The same question gets very different answers depending on the persona
    you assign in the system prompt.
    """

    question = "How should I handle schema changes in my data pipeline?"

    personas = {
        "No system prompt": None,
        "Data engineer": "You are a senior data engineer with 10 years experience using Airflow, dbt, and Snowflake. Give practical, opinionated advice.",
        "Cautious architect": "You are a data architect focused on enterprise governance and backward compatibility. Prioritise stability over speed.",
    }

    print("=== Role Framing Demo ===\n")

    for name, system in personas.items():
        kwargs = dict(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": question}]
        )
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        print(f"Persona: {name}")
        print(response.content[0].text[:300].strip())
        print()


# ── Demo 2: Output contract — enforce JSON schema ─────────────────────────────
def output_contract_demo():
    """
    Defining the exact output shape in the system prompt is the most
    reliable way to get consistent structured output. No parsing gymnastics.
    """

    system = """You are a data quality analyst.
When given a dataset description, respond ONLY with valid JSON — no markdown,
no explanation, no preamble. Use exactly this schema:
{
  "overall_score": 0-100,
  "issues": [
    {"field": "column name", "issue": "description", "severity": "high|medium|low"}
  ],
  "recommendation": "one sentence action"
}"""

    datasets = [
        "orders table: 15% null customer_id, 3% duplicate order_ids, created_at has dates from 1970",
        "products table: all columns populated, no duplicates, prices all positive",
    ]

    print("=== Output Contract Demo ===\n")

    for dataset in datasets:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": f"Analyse: {dataset}"}]
        )
        result = json.loads(response.content[0].text)
        print(f"Dataset: {dataset[:60]}...")
        print(json.dumps(result, indent=2))
        print()


# ── Demo 3: Constraints — safety rails for production ─────────────────────────
def constraints_demo():
    """
    Use the system prompt to add guardrails. For a SQL assistant, you
    might forbid destructive operations so no one accidentally drops a table.
    """

    system = """You are a SQL assistant for our Snowflake data warehouse.
CONSTRAINTS (never violate these):
- Only write SELECT statements. Never write INSERT, UPDATE, DELETE, DROP, TRUNCATE, or ALTER.
- Always include a LIMIT clause (max 10000) unless the user explicitly says "no limit".
- Always use fully qualified table names: database.schema.table
- If asked to do something that violates these rules, explain why you can't and suggest an alternative.

Our database: PROD_DW. Main schemas: RAW, STAGING, MARTS."""

    requests = [
        "Show me the top 10 customers by revenue",
        "Delete all orders older than 2020",
        "Give me all rows from the orders table",
    ]

    print("=== Constraints Demo ===\n")

    for req in requests:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": req}]
        )
        print(f"Request: '{req}'")
        print(response.content[0].text.strip()[:300])
        print()


# ── Demo 4: Few-shot examples in system prompt ────────────────────────────────
def few_shot_demo():
    """
    Including 2-3 examples of ideal input/output in the system prompt
    dramatically improves consistency for structured tasks.
    """

    system = """You are a dbt model namer. Given a description of what a model does,
return the correct snake_case dbt model name following our naming conventions:
- stg_ prefix for staging models
- fct_ prefix for fact tables
- dim_ prefix for dimension tables
- int_ prefix for intermediate models

Examples:
Input: "Cleans and standardises raw Stripe payment events"
Output: stg_stripe__payments

Input: "One row per order with revenue metrics"
Output: fct_orders

Input: "All customer attributes including lifetime value"
Output: dim_customers

Respond with ONLY the model name, nothing else."""

    descriptions = [
        "Raw Fivetran Salesforce opportunity records, lightly cleaned",
        "Daily revenue aggregated by product category and region",
        "Intermediate model joining orders with their line items",
        "All product attributes including current price and category",
    ]

    print("=== Few-Shot Examples Demo ===\n")

    for desc in descriptions:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=32,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": desc}]
        )
        print(f"Description: {desc}")
        print(f"Model name:  {response.content[0].text.strip()}\n")


# ── Demo 5: Domain context injection ─────────────────────────────────────────
def domain_context_demo():
    """
    Inject your actual schema, conventions, and stack into the system prompt.
    The model then uses this context to give answers specific to YOUR setup.
    """

    # In production this would be loaded from your dbt docs or schema registry
    our_schema = """
    Key tables (Snowflake, PROD_DW):
    - MARTS.FCT_ORDERS: grain=order_id, contains order_total, customer_id, created_at, status
    - MARTS.DIM_CUSTOMERS: grain=customer_id, contains region, segment, signup_date
    - MARTS.FCT_REVENUE: grain=date+product+region, contains net_revenue, gross_revenue
    - STAGING.STG_STRIPE__PAYMENTS: raw Stripe data, refreshed hourly via Fivetran

    Conventions:
    - All timestamps are UTC
    - Deleted records use is_deleted=true (soft deletes)
    - Partition key is always created_at truncated to day
    """

    system = f"""You are a data analyst assistant for our company.
You have deep knowledge of our data warehouse schema.

<schema>
{our_schema}
</schema>

Always use the exact table and column names from the schema above."""

    questions = [
        "How do I get total revenue by region for last month?",
        "Which table should I use to find when a customer first signed up?",
    ]

    print("=== Domain Context Demo ===\n")

    for q in questions:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": q}]
        )
        print(f"Q: {q}")
        print(f"A: {response.content[0].text.strip()}\n")


if __name__ == "__main__":
    role_framing_demo()
    output_contract_demo()
    constraints_demo()
    few_shot_demo()
    domain_context_demo()
