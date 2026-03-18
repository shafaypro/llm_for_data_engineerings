"""
09_hallucination_and_evals_anthropic.py
========================================
Concept: Hallucination & Evals
-------------------------------
Hallucination = the model confidently generates plausible-but-wrong output.
Unlike a SQL error that throws an exception, hallucination is silent.
You must build validation and eval layers to catch it.

Evals = automated tests for LLM outputs. Same mindset as dbt tests:
define what good output looks like, run it on every change, alert on regression.

Mitigation layers:
1. System prompt: "say I don't know if unsure"
2. RAG: ground answers in real documents
3. Output validation: parse + schema check + semantic check
4. LLM-as-judge: use another LLM call to score the first one
5. Eval suite: regression tests that run on every prompt change

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json
import re
from typing import Optional

client = anthropic.Anthropic()


# ── Demo 1: Hallucination without mitigations ─────────────────────────────────
def hallucination_demo():
    """
    Ask the model about internal company data with no context.
    It will confidently make something up.
    This is the baseline — what you get with NO mitigations.
    """

    print("=== Hallucination Demo (no mitigations) ===\n")

    # The model has no idea about your company's internals
    questions = [
        "What was the total revenue of our orders pipeline yesterday?",
        "Who is the on-call engineer this week?",
        "What is the SLA for the fct_orders dbt model?",
    ]

    for q in questions:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": q}]
        )
        print(f"Q: {q}")
        print(f"A (⚠ hallucinated): {response.content[0].text[:200].strip()}\n")


# ── Demo 2: Mitigation — "say I don't know" in system prompt ──────────────────
def idontknow_mitigation_demo():
    """
    Simple but effective: explicitly tell the model to admit uncertainty.
    Without grounding data (RAG), this is your first line of defence.
    """

    print("=== I-Don't-Know Mitigation Demo ===\n")

    system = """You are a data engineering assistant.
IMPORTANT: You only have access to information provided in this conversation.
If asked about specific metrics, data, or internal company information that
has not been provided to you, say exactly: "I don't have that information —
please query the data warehouse directly or check the runbook."
Never make up numbers, names, or dates."""

    questions = [
        "What was yesterday's revenue?",
        "What is a window function in SQL?",  # general knowledge — fine to answer
    ]

    for q in questions:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=system,
            messages=[{"role": "user", "content": q}]
        )
        print(f"Q: {q}")
        print(f"A: {response.content[0].text.strip()}\n")


# ── Demo 3: Output validation ─────────────────────────────────────────────────
def output_validation_demo():
    """
    For structured outputs (SQL, JSON, column names), validate programmatically
    before using the result. Never blindly execute LLM-generated SQL.
    """

    print("=== Output Validation Demo ===\n")

    def generate_and_validate_sql(question: str, known_tables: list) -> dict:
        """Generate SQL and validate it before returning."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            temperature=0,
            system="""Generate a Snowflake SQL SELECT query. Return ONLY the SQL, no explanation.
Available tables: orders(order_id, customer_id, created_at, total_amount, status),
customers(customer_id, name, region, segment)""",
            messages=[{"role": "user", "content": question}]
        )

        sql = response.content[0].text.strip()

        # Validation checks
        issues = []

        # 1. Must be a SELECT statement
        if not sql.upper().strip().startswith("SELECT"):
            issues.append("Not a SELECT statement — dangerous query rejected")

        # 2. No destructive keywords
        dangerous = ["DELETE", "DROP", "TRUNCATE", "INSERT", "UPDATE", "ALTER", "CREATE"]
        for keyword in dangerous:
            if keyword in sql.upper():
                issues.append(f"Dangerous keyword '{keyword}' detected")

        # 3. Check table names are real
        for table in known_tables:
            if table.lower() in sql.lower():
                break
        else:
            issues.append(f"Query references unknown tables (known: {known_tables})")

        # 4. Check it has a LIMIT (safety for runaway queries)
        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + "\nLIMIT 1000;"
            issues.append("Added LIMIT 1000 as safety measure")

        return {
            "sql": sql,
            "valid": len([i for i in issues if "dangerous" in i.lower() or "Not a SELECT" in i]) == 0,
            "issues": issues
        }

    questions = [
        "Show me the top 10 customers by total order value",
        "Delete all orders from 2020",  # dangerous — should be blocked
        "Get revenue by region for the last 7 days",
    ]

    for q in questions:
        result = generate_and_validate_sql(q, known_tables=["orders", "customers"])
        print(f"Question: {q}")
        print(f"Valid: {result['valid']}")
        if result['issues']:
            print(f"Issues: {result['issues']}")
        if result['valid']:
            print(f"SQL:\n{result['sql']}")
        print()


# ── Demo 4: LLM-as-judge eval ─────────────────────────────────────────────────
def llm_as_judge_demo():
    """
    Use a second LLM call to evaluate the quality of the first.
    This is the standard technique for evaluating open-ended outputs
    like summaries, explanations, and reports where exact-match fails.
    """

    print("=== LLM-as-Judge Eval Demo ===\n")

    def generate_answer(question: str) -> str:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": question}]
        )
        return response.content[0].text.strip()

    def judge_answer(question: str, answer: str, criteria: list) -> dict:
        """Use Claude to score an answer on a set of criteria."""
        criteria_str = "\n".join([f"- {c}" for c in criteria])

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            temperature=0,
            system="""You are an objective evaluator. Score answers honestly.
Respond ONLY with valid JSON — no explanation outside the JSON.""",
            messages=[{"role": "user", "content": f"""Evaluate this answer.

Question: {question}
Answer: {answer}

Score each criterion from 1-5 (5=excellent) and give a brief reason:
{criteria_str}

Return JSON:
{{"scores": {{"criterion_name": {{"score": 1-5, "reason": "brief explanation"}}}}, "overall": 1-5}}"""}]
        )
        return json.loads(response.content[0].text)

    # Test a data engineering question
    question = "How do I handle duplicate records when loading data into Snowflake?"
    answer = generate_answer(question)

    print(f"Question: {question}")
    print(f"Answer: {answer[:300]}...")
    print()

    scores = judge_answer(question, answer, criteria=[
        "accuracy: Is the information technically correct?",
        "actionability: Does it give concrete steps an engineer can follow?",
        "completeness: Does it cover the main approaches?",
    ])

    print("Evaluation scores:")
    for criterion, result in scores["scores"].items():
        bar = "★" * result["score"] + "☆" * (5 - result["score"])
        print(f"  {criterion}: {bar} ({result['score']}/5) — {result['reason']}")
    print(f"  Overall: {scores['overall']}/5")


# ── Demo 5: Regression eval suite ─────────────────────────────────────────────
def regression_eval_suite():
    """
    A mini eval suite — the equivalent of dbt tests for your LLM pipeline.
    Run this after every prompt change to catch regressions.
    In production, integrate with pytest and run in CI.
    """

    print("\n=== Regression Eval Suite ===\n")

    # Define test cases: (input, expected properties of the output)
    test_cases = [
        {
            "name": "JSON output is valid",
            "input": "Classify this alert as P1/P2/P3 and category: 'Snowflake credits exhausted'",
            "system": 'Respond ONLY with JSON: {"severity": "P1|P2|P3", "category": "string"}',
            "check": lambda output: json.loads(output) and "severity" in json.loads(output),
            "error": "Output is not valid JSON with 'severity' key"
        },
        {
            "name": "SQL starts with SELECT",
            "input": "Write SQL to count orders today",
            "system": "Write a SQL query. Return ONLY the SQL, nothing else.",
            "check": lambda output: output.strip().upper().startswith("SELECT"),
            "error": "SQL does not start with SELECT"
        },
        {
            "name": "No hallucinated company data",
            "input": "What was our revenue yesterday?",
            "system": "You are a data assistant. Only answer from provided context. If no context, say you don't know.",
            "check": lambda output: "don't" in output.lower() or "do not" in output.lower() or "no information" in output.lower(),
            "error": "Model may have hallucinated company-specific data"
        },
        {
            "name": "Response is concise (under 200 words)",
            "input": "What is a foreign key?",
            "system": "Be very concise. Answer in 1-2 sentences max.",
            "check": lambda output: len(output.split()) < 200,
            "error": "Response exceeds 200 words — may be ignoring conciseness instruction"
        },
    ]

    passed = 0
    for test in test_cases:
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                temperature=0,
                system=test["system"],
                messages=[{"role": "user", "content": test["input"]}]
            )
            output = response.content[0].text.strip()
            result = test["check"](output)
            status = "PASS" if result else "FAIL"
            if result:
                passed += 1
        except Exception as e:
            status = "FAIL"
            output = str(e)

        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {test['name']}: {status}")
        if status == "FAIL":
            print(f"    Error: {test['error']}")
            print(f"    Output: {output[:100]}")

    print(f"\n{passed}/{len(test_cases)} tests passed")


if __name__ == "__main__":
    hallucination_demo()
    idontknow_mitigation_demo()
    output_validation_demo()
    llm_as_judge_demo()
    regression_eval_suite()
