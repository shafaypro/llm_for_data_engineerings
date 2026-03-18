"""
07_tool_use_anthropic.py
========================
Concept: Tool Use / Function Calling
-------------------------------------
Tool use lets Claude decide at runtime which functions to call and with
what arguments. You define the tools; Claude decides the strategy.
This is the core pattern behind every AI agent.

Think of it like an Airflow task that can branch — the LLM is the
branch operator, but instead of a fixed condition, it reasons about
what to do next based on the results it gets back.

The loop:
  1. Send user message + tool definitions to Claude
  2. Claude responds with a tool_use block (name + arguments)
  3. Your code executes the tool and gets a result
  4. Send the result back to Claude
  5. Claude either calls another tool or gives a final answer
  6. Repeat until stop_reason == "end_turn"

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json
from datetime import datetime, timedelta
import random

client = anthropic.Anthropic()


# ── Tool definitions (what Claude can see and call) ────────────────────────────

TOOLS = [
    {
        "name": "run_sql",
        "description": "Execute a read-only SQL query against the Snowflake data warehouse and return results as JSON. Only SELECT statements are allowed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A valid SQL SELECT statement"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return. Default 100, max 1000.",
                    "default": 100
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_dag_status",
        "description": "Get the most recent run status of an Airflow DAG including run time, failed tasks, and error messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dag_id": {
                    "type": "string",
                    "description": "The Airflow DAG ID (e.g. 'daily_orders_etl')"
                }
            },
            "required": ["dag_id"]
        }
    },
    {
        "name": "get_dbt_model_info",
        "description": "Get metadata about a dbt model including its description, columns, tests, and last run status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "The dbt model name (e.g. 'fct_orders')"
                }
            },
            "required": ["model_name"]
        }
    },
    {
        "name": "trigger_dag",
        "description": "Trigger an Airflow DAG run manually. Only use this if explicitly asked by the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dag_id": {"type": "string"},
                "conf": {
                    "type": "object",
                    "description": "Optional run configuration as JSON",
                    "default": {}
                }
            },
            "required": ["dag_id"]
        }
    }
]


# ── Mock tool implementations (replace with real calls in production) ──────────

def run_sql(query: str, limit: int = 100) -> str:
    """Mock SQL execution. Replace with: snowflake.connector or sqlalchemy."""
    query_lower = query.lower()

    if "revenue" in query_lower or "amount" in query_lower:
        return json.dumps({
            "rows": [
                {"date": "2025-03-17", "revenue": 142_530.50, "orders": 312},
                {"date": "2025-03-16", "revenue": 138_920.00, "orders": 298},
                {"date": "2025-03-15", "revenue": 151_200.75, "orders": 334},
            ],
            "row_count": 3,
            "query_time_ms": 423
        })
    elif "customer" in query_lower:
        return json.dumps({
            "rows": [
                {"customer_id": "C001", "name": "Acme Corp", "total_revenue": 45_200},
                {"customer_id": "C002", "name": "Globex Ltd", "total_revenue": 38_100},
            ],
            "row_count": 2,
            "query_time_ms": 287
        })
    else:
        return json.dumps({"rows": [], "row_count": 0, "message": "No results found"})


def get_dag_status(dag_id: str) -> str:
    """Mock Airflow API. Replace with: requests.get(airflow_url/api/v1/dags/{dag_id}/runs)"""
    statuses = {
        "daily_orders_etl": {
            "dag_id": dag_id,
            "last_run_id": "scheduled__2025-03-17T02:00:00+00:00",
            "state": "failed",
            "start_date": "2025-03-17T02:00:12Z",
            "end_date": "2025-03-17T02:23:45Z",
            "duration_minutes": 23,
            "failed_task": "load_to_snowflake",
            "error": "OperationalError: Snowflake connection timeout after 300s. Warehouse TRANSFORM_WH may be suspended.",
        },
        "daily_revenue_report": {
            "dag_id": dag_id,
            "last_run_id": "scheduled__2025-03-17T03:00:00+00:00",
            "state": "success",
            "start_date": "2025-03-17T03:00:05Z",
            "end_date": "2025-03-17T03:18:22Z",
            "duration_minutes": 18,
        }
    }
    return json.dumps(statuses.get(dag_id, {"dag_id": dag_id, "state": "not_found"}))


def get_dbt_model_info(model_name: str) -> str:
    """Mock dbt metadata. Replace with: dbt Cloud API or dbt artifacts (manifest.json)."""
    models = {
        "fct_orders": {
            "model": model_name,
            "description": "One row per order. Core fact table for all order analytics.",
            "columns": ["order_id", "customer_id", "order_date", "total_amount", "net_amount", "status", "region"],
            "tests": ["not_null(order_id)", "unique(order_id)", "accepted_values(status)"],
            "last_run": "2025-03-17 02:45:00",
            "last_run_status": "success",
            "row_count": 4_823_112,
            "depends_on": ["stg_orders", "dim_customers"]
        }
    }
    return json.dumps(models.get(model_name, {"model": model_name, "error": "Model not found"}))


def trigger_dag(dag_id: str, conf: dict = None) -> str:
    """Mock DAG trigger. Replace with: requests.post(airflow_url/api/v1/dags/{dag_id}/dagRuns)"""
    run_id = f"manual__{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')}+00:00"
    return json.dumps({
        "dag_id": dag_id,
        "run_id": run_id,
        "state": "queued",
        "message": f"DAG '{dag_id}' triggered successfully. Run ID: {run_id}"
    })


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict) -> str:
    """Route tool calls to their implementations."""
    tools = {
        "run_sql": lambda i: run_sql(i["query"], i.get("limit", 100)),
        "get_dag_status": lambda i: get_dag_status(i["dag_id"]),
        "get_dbt_model_info": lambda i: get_dbt_model_info(i["model_name"]),
        "trigger_dag": lambda i: trigger_dag(i["dag_id"], i.get("conf", {})),
    }
    handler = tools.get(name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return handler(inputs)


# ── The agent loop ─────────────────────────────────────────────────────────────

def run_agent(user_question: str, verbose: bool = True) -> str:
    """
    Core agentic loop. Claude calls tools until it has enough info to answer.
    This is the same pattern used by LangChain, CrewAI, LlamaIndex — just
    written explicitly so you can see exactly what's happening.
    """

    messages = [{"role": "user", "content": user_question}]

    system = """You are a data engineering assistant with access to our data infrastructure.
You can query the data warehouse, check Airflow DAG statuses, and inspect dbt models.

Guidelines:
- Use tools to get accurate, real-time information before answering
- If a DAG is failing, diagnose the root cause using available tools
- Be concise and actionable in your final answer
- Only trigger DAGs if the user explicitly asks you to"""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {user_question}")
        print(f"{'='*60}")

    iteration = 0
    max_iterations = 10  # safety limit to prevent infinite loops

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages
        )

        # Collect tool calls from this response
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if verbose and text_blocks:
            for block in text_blocks:
                if block.text.strip():
                    print(f"\n[Claude thinking]: {block.text[:200]}")

        # No more tool calls — Claude has its final answer
        if response.stop_reason == "end_turn":
            final_text = next((b.text for b in response.content if b.type == "text"), "")
            if verbose:
                print(f"\n[Final Answer]:\n{final_text}")
            return final_text

        # Execute all tool calls in this round
        tool_results = []
        for tool_call in tool_calls:
            if verbose:
                print(f"\n  → Tool: {tool_call.name}({json.dumps(tool_call.input)})")

            result = execute_tool(tool_call.name, tool_call.input)

            if verbose:
                result_preview = result[:200] + "..." if len(result) > 200 else result
                print(f"  ← Result: {result_preview}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result
            })

        # Add Claude's response + tool results to conversation history
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached. Please try a more specific question."


# ── Demo questions ─────────────────────────────────────────────────────────────

def main():
    questions = [
        "What's today's revenue compared to yesterday?",
        "The daily_orders_etl DAG is failing — can you diagnose what's wrong?",
        "Tell me about the fct_orders model and how many rows it has.",
    ]

    for question in questions:
        run_agent(question, verbose=True)
        print("\n")


if __name__ == "__main__":
    main()
