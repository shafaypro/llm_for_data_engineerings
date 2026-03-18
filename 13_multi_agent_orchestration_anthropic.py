"""
13_multi_agent_orchestration_anthropic.py
==========================================
Concept: Multi-Agent Systems
-----------------------------
Instead of one LLM doing everything, you split work across specialised agents.
Each agent has a focused role, its own tools, and its own system prompt.
An orchestrator agent coordinates them — deciding which specialist to call
and synthesising their outputs.

Think of it like a microservices architecture for AI:
- Monolith agent = one big prompt trying to do everything (fragile, hard to debug)
- Multi-agent = each agent is a focused service with a clear contract

Patterns covered:
1. Orchestrator + specialist agents (most common)
2. Sequential pipeline (agent A output → agent B input)
3. Parallel fan-out + merge
4. Critic / reviewer agent (self-refinement)

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

client = anthropic.Anthropic()


# ── Shared agent runner ────────────────────────────────────────────────────────

def run_agent(role: str, task: str, context: str = "", model: str = "claude-haiku-4-5-20251001",
              max_tokens: int = 512, temperature: float = 0) -> str:
    """Generic single-shot agent call."""
    messages = [{"role": "user", "content": f"{f'<context>{context}</context>' if context else ''}\n\n{task}"}]
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=role,
        messages=messages
    )
    return response.content[0].text.strip()


# ── Pattern 1: Orchestrator + specialists ─────────────────────────────────────

def orchestrator_pattern():
    """
    An orchestrator receives the user request, breaks it into subtasks,
    routes each to the right specialist, then synthesises the results.

    Use case: 'Investigate why revenue dropped 30% yesterday'
    - SQL agent queries the data
    - Pipeline agent checks DAG health
    - Analyst agent synthesises findings into a report
    """

    print("=== Pattern 1: Orchestrator + Specialists ===\n")

    # Specialist agents
    AGENTS = {
        "sql_analyst": {
            "role": "You are a SQL data analyst. Given a business question, write and interpret SQL queries against Snowflake. Tables: fct_orders(order_id, customer_id, order_date, total_amount, status, region), dim_customers(customer_id, segment, region). Return findings as JSON.",
            "mock_result": json.dumps({"finding": "Revenue dropped from $145k to $98k. EMEA region fell 52%. Enterprise segment down 61%. Order count unchanged — AOV dropped.", "sql": "SELECT region, segment, SUM(total_amount) FROM fct_orders GROUP BY 1,2"})
        },
        "pipeline_inspector": {
            "role": "You are a data pipeline reliability engineer. Analyse pipeline health metrics and identify issues. Return findings as JSON.",
            "mock_result": json.dumps({"finding": "fct_orders dbt model ran 3.5 hours late yesterday. Root cause: Snowflake warehouse auto-suspended during peak load. Data was complete but landed at 11am instead of 7am.", "severity": "medium"})
        },
        "report_writer": {
            "role": "You are a senior data analyst writing executive summaries. Given findings from multiple sources, write a clear, concise incident report. Use plain English, no jargon. Format as markdown.",
            "mock_result": None  # will actually call Claude for this one
        }
    }

    incident = "Revenue dropped 30% yesterday compared to same day last week. Investigate and write a summary."

    print(f"Incident: {incident}\n")

    # Step 1: Orchestrator plans the investigation
    plan = run_agent(
        role="You are a data engineering team lead. Break down incidents into investigation tasks. Return JSON: {\"tasks\": [{\"agent\": \"sql_analyst|pipeline_inspector\", \"task\": \"specific question\"}]}",
        task=incident,
        model="claude-haiku-4-5-20251001",
        max_tokens=256
    )

    try:
        tasks = json.loads(plan)["tasks"]
    except Exception:
        tasks = [
            {"agent": "sql_analyst", "task": "What changed in revenue metrics yesterday vs last week?"},
            {"agent": "pipeline_inspector", "task": "Were there any pipeline delays or failures yesterday?"}
        ]

    print(f"Orchestrator planned {len(tasks)} tasks:")
    for t in tasks:
        print(f"  → {t['agent']}: {t['task']}")

    # Step 2: Run specialist agents (using mock results for demo)
    findings = {}
    for task in tasks:
        agent_name = task["agent"]
        agent = AGENTS.get(agent_name, AGENTS["sql_analyst"])
        print(f"\nRunning {agent_name}...")
        # In production: actually call Claude for each specialist
        # findings[agent_name] = run_agent(role=agent["role"], task=task["task"])
        findings[agent_name] = agent["mock_result"]  # mock for demo
        print(f"  Result: {findings[agent_name][:100]}...")

    # Step 3: Report writer synthesises all findings
    print("\nReport writer synthesising findings...")
    context = "\n\n".join([f"[{k}]\n{v}" for k, v in findings.items()])
    report = run_agent(
        role=AGENTS["report_writer"]["role"],
        task=f"Write an incident report for: {incident}",
        context=context,
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        temperature=0.3
    )

    print(f"\n{'='*50}")
    print("INCIDENT REPORT:")
    print(report)


# ── Pattern 2: Sequential pipeline ────────────────────────────────────────────

def sequential_pipeline_pattern():
    """
    Agent A's output becomes Agent B's input. Each agent does one thing well.
    Use case: raw log → structured JSON → Slack message
    """

    print("\n=== Pattern 2: Sequential Pipeline ===\n")

    raw_log = """
    [2025-03-17 02:23:41] ERROR - Task load_to_snowflake on DagRun daily_orders_etl
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/site-packages/snowflake/connector/cursor.py", line 912
    snowflake.connector.errors.ProgrammingError: 090114 (22000): Numeric value '2025-13-45'
    is not recognized. Row 8823, Column 'order_date'.
    Task exited with return code 1
    """

    # Stage 1: Extract structured error info
    print("Stage 1: Extracting error info...")
    structured = run_agent(
        role="You are a log parser. Extract error info from logs. Return ONLY JSON: {\"error_type\": \"\", \"table\": \"\", \"row\": 0, \"column\": \"\", \"value\": \"\", \"dag_id\": \"\"}",
        task=f"Parse this log:\n{raw_log}",
        temperature=0
    )
    print(f"  Structured: {structured}")

    # Stage 2: Generate fix suggestion
    print("\nStage 2: Generating fix...")
    fix = run_agent(
        role="You are a data quality expert. Given a structured error, provide a specific fix in 2 sentences max.",
        task="What is the fix for this error?",
        context=structured,
        temperature=0
    )
    print(f"  Fix: {fix}")

    # Stage 3: Write Slack message
    print("\nStage 3: Writing Slack alert...")
    slack_msg = run_agent(
        role="You are writing a Slack alert for a data engineering team. Be concise. Use this format: *DAG* | *Error* | *Fix*. No markdown code blocks.",
        task="Write the Slack alert.",
        context=f"Error info: {structured}\nSuggested fix: {fix}",
        temperature=0.2,
        max_tokens=100
    )
    print(f"  Slack: {slack_msg}")


# ── Pattern 3: Parallel fan-out ────────────────────────────────────────────────

def parallel_fanout_pattern():
    """
    Run multiple agents concurrently and merge results.
    Use case: check data quality across multiple tables simultaneously.
    Much faster than sequential when tasks are independent.
    """

    print("\n=== Pattern 3: Parallel Fan-out ===\n")

    tables = {
        "fct_orders": "10% null customer_id, 2% duplicate order_ids",
        "dim_customers": "All fields populated, no duplicates detected",
        "fct_revenue": "3 rows with negative net_revenue, date gaps on weekends",
    }

    def check_table(table_name: str, stats: str) -> dict:
        result = run_agent(
            role='You are a data quality analyst. Score tables 0-100. Return JSON: {"table": "", "score": 0, "issues": [], "priority": "low|medium|high"}',
            task=f"Assess data quality for table {table_name}: {stats}",
            temperature=0,
            max_tokens=128
        )
        try:
            return json.loads(result)
        except Exception:
            return {"table": table_name, "score": 50, "issues": [result], "priority": "medium"}

    print(f"Checking {len(tables)} tables in parallel...\n")
    start_results = {}

    # Run all checks concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(check_table, name, stats): name
                   for name, stats in tables.items()}
        for future in as_completed(futures):
            result = future.result()
            start_results[result["table"]] = result
            print(f"  ✓ {result['table']}: score={result['score']}, priority={result['priority']}")

    # Merge: orchestrator summarises all results
    all_results = json.dumps(list(start_results.values()), indent=2)
    summary = run_agent(
        role="You are a data engineering lead. Summarise data quality results in 2 sentences. Prioritise what needs immediate attention.",
        task="Summarise these data quality results.",
        context=all_results,
        temperature=0.3,
        max_tokens=150
    )
    print(f"\nSummary: {summary}")


# ── Pattern 4: Critic / self-refinement ────────────────────────────────────────

def critic_pattern():
    """
    A generator agent produces output. A critic agent reviews it.
    If the critic scores it below a threshold, the generator tries again
    with the critic's feedback. This improves output quality without
    manual prompt iteration.

    Use case: generating dbt model SQL that must be correct AND efficient.
    """

    print("\n=== Pattern 4: Critic / Self-Refinement ===\n")

    task = "Write a Snowflake SQL query that finds customers who placed orders in Q1 2025 but not in Q2 2025 (churned customers). Tables: fct_orders(order_id, customer_id, order_date), dim_customers(customer_id, name, segment)."

    for attempt in range(1, 4):  # max 3 attempts
        print(f"Attempt {attempt}: Generating SQL...")

        sql = run_agent(
            role="You are a Snowflake SQL expert. Write efficient, correct SQL. Return ONLY the SQL query, no explanation.",
            task=task,
            temperature=0,
            max_tokens=300
        )
        print(f"  Generated:\n  {sql[:200]}...")

        # Critic evaluates the SQL
        critique = run_agent(
            role='You are a SQL code reviewer. Score SQL from 0-100. Return JSON: {"score": 0-100, "issues": ["list issues"], "approved": true/false, "feedback": "specific improvement"}',
            task=f"Review this SQL for the task: {task}\n\nSQL:\n{sql}",
            temperature=0,
            max_tokens=200
        )

        try:
            review = json.loads(critique)
        except Exception:
            review = {"score": 80, "approved": True, "issues": [], "feedback": ""}

        print(f"  Critic score: {review['score']}/100 | Approved: {review['approved']}")

        if review.get("approved") or review.get("score", 0) >= 85:
            print(f"  ✓ Approved on attempt {attempt}")
            print(f"\nFinal SQL:\n{sql}")
            break
        else:
            print(f"  Issues: {review.get('issues', [])}")
            print(f"  Feedback: {review.get('feedback', '')}")
            # Incorporate feedback into next attempt
            task = f"{task}\n\nPrevious attempt had these issues: {review.get('feedback', '')}. Fix them."
    else:
        print("  Max attempts reached — using last version")


if __name__ == "__main__":
    orchestrator_pattern()
    sequential_pipeline_pattern()
    parallel_fanout_pattern()
    critic_pattern()
