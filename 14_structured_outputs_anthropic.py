"""
14_structured_outputs_anthropic.py
====================================
Concept: Structured Outputs
----------------------------
Getting reliable, typed, validated JSON from Claude every time.
Critical for production pipelines — you can't afford to have a model
return free-text when downstream code expects {"severity": "P1"}.

Techniques from weakest to strongest:
1. System prompt JSON contract (basic, works well with good prompts)
2. JSON mode via prefill (force the model to start with "{")
3. Pydantic validation + retry loop (production-grade)
4. Tool use as structured output (most reliable — JSON schema enforced)

Install: pip install anthropic pydantic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json
from typing import List, Optional, Literal
from pydantic import BaseModel, ValidationError, field_validator

client = anthropic.Anthropic()


# ── Pydantic models — define your output schemas here ─────────────────────────

class PipelineAlert(BaseModel):
    severity: Literal["P1", "P2", "P3"]
    category: Literal["infra", "data_quality", "performance", "auth"]
    summary: str
    suggested_fix: str
    is_flaky: bool

    @field_validator("summary", "suggested_fix")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class DbtModelDocs(BaseModel):
    model_description: str
    columns: dict[str, str]

    @field_validator("model_description")
    @classmethod
    def description_min_length(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("Description too short")
        return v


class DataQualityReport(BaseModel):
    table_name: str
    overall_score: int
    issues: List[str]
    recommendation: str
    action_required: bool

    @field_validator("overall_score")
    @classmethod
    def score_range(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("Score must be 0-100")
        return v


class SQLAnalysis(BaseModel):
    query_type: Literal["SELECT", "DML", "DDL", "UNKNOWN"]
    tables_referenced: List[str]
    has_where_clause: bool
    has_limit: bool
    estimated_complexity: Literal["simple", "moderate", "complex"]
    potential_issues: List[str]
    optimised_query: Optional[str] = None


# ── Technique 1: System prompt JSON contract ──────────────────────────────────

def technique_1_system_prompt():
    """Basic: describe the JSON shape in the system prompt."""
    print("=== Technique 1: System Prompt JSON Contract ===\n")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        temperature=0,
        system="""You are a pipeline alert classifier.
Respond ONLY with valid JSON — no markdown, no preamble, no trailing text.
Schema: {"severity": "P1|P2|P3", "category": "infra|data_quality|performance|auth",
"summary": "string", "suggested_fix": "string", "is_flaky": boolean}""",
        messages=[{"role": "user", "content": "Alert: Snowflake warehouse auto-suspended, orders DAG failed after 3 retries"}]
    )

    raw = response.content[0].text.strip()
    print(f"Raw output: {raw}")
    try:
        alert = PipelineAlert(**json.loads(raw))
        print(f"Validated: severity={alert.severity}, category={alert.category}")
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Validation failed: {e}")


# ── Technique 2: Prefill — force JSON start ────────────────────────────────────

def technique_2_prefill():
    """
    Prefill the assistant turn with '{' to force the model to continue
    generating valid JSON. Highly reliable for preventing preamble like
    'Here is the JSON you requested: ...'
    """
    print("\n=== Technique 2: Prefill to Force JSON ===\n")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        temperature=0,
        system="You are a data quality analyst. Return JSON only.",
        messages=[
            {"role": "user", "content": "Analyse: fct_orders has 8% null customer_id and 200 duplicate order_ids"},
            {"role": "assistant", "content": "{"}  # <-- prefill forces JSON continuation
        ]
    )

    # Prepend the opening brace we used as prefill
    raw = "{" + response.content[0].text.strip()
    print(f"Raw: {raw[:200]}")
    try:
        data = json.loads(raw)
        print(f"Parsed keys: {list(data.keys())}")
    except json.JSONDecodeError as e:
        print(f"Parse error: {e}")


# ── Technique 3: Pydantic validation + retry loop ─────────────────────────────

def technique_3_pydantic_retry():
    """
    Production-grade: try to parse + validate, and if it fails,
    send the error back to Claude and ask it to fix the output.
    This handles the ~2% of cases where the model goes slightly off-schema.
    """
    print("\n=== Technique 3: Pydantic Validation + Retry ===\n")

    def call_with_validation(prompt: str, schema: type[BaseModel],
                              max_retries: int = 3) -> BaseModel:
        system = f"""Return ONLY valid JSON matching this Pydantic schema:
{schema.model_json_schema()}
No markdown. No explanation. Just the JSON object."""

        messages = [{"role": "user", "content": prompt}]
        last_error = None

        for attempt in range(1, max_retries + 1):
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                temperature=0,
                system=system,
                messages=messages
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            try:
                data = json.loads(raw)
                validated = schema(**data)
                print(f"  Attempt {attempt}: SUCCESS")
                return validated
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                print(f"  Attempt {attempt}: FAILED — {last_error[:80]}")
                # Send error back to Claude for correction
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"That output failed validation: {last_error}\nFix it and return only the corrected JSON."})

        raise ValueError(f"Failed after {max_retries} attempts. Last error: {last_error}")

    # Test with different schemas
    test_cases = [
        ("Classify: 'S3 bucket access denied for data lake'", PipelineAlert),
        ("Document dbt model: fct_revenue with columns: revenue_date, region, net_revenue, gross_revenue", DbtModelDocs),
        ("Assess: orders table has 5% nulls in customer_id, no duplicates", DataQualityReport),
    ]

    for prompt, schema in test_cases:
        print(f"\nSchema: {schema.__name__}")
        print(f"Prompt: {prompt[:60]}...")
        try:
            result = call_with_validation(prompt, schema)
            print(f"Result: {result.model_dump()}")
        except ValueError as e:
            print(f"Error: {e}")


# ── Technique 4: Tool use as structured output (most reliable) ────────────────

def technique_4_tool_use():
    """
    Define the desired output as a 'tool' with a strict JSON schema.
    The model MUST call the tool with valid arguments — no free-text escape hatch.
    This is the most reliable way to get structured output from Claude.

    Pattern: define a 'submit_result' tool that accepts your schema.
    Force tool use with tool_choice={'type': 'tool', 'name': 'submit_result'}.
    """
    print("\n=== Technique 4: Tool Use as Structured Output ===\n")

    # Define the output schema as a tool
    output_tool = {
        "name": "submit_alert_analysis",
        "description": "Submit the structured analysis of a pipeline alert",
        "input_schema": {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3"],
                    "description": "Alert severity level"
                },
                "category": {
                    "type": "string",
                    "enum": ["infra", "data_quality", "performance", "auth"],
                },
                "summary": {"type": "string"},
                "suggested_fix": {"type": "string"},
                "is_flaky": {"type": "boolean"}
            },
            "required": ["severity", "category", "summary", "suggested_fix", "is_flaky"]
        }
    }

    alerts = [
        "CRITICAL: fct_orders dbt model has been running for 4 hours, normally takes 20 minutes",
        "ERROR: Fivetran Salesforce connector authentication token expired",
        "WARNING: 15% of rows in dim_customers have null email addresses",
    ]

    for alert in alerts:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            tools=[output_tool],
            tool_choice={"type": "tool", "name": "submit_alert_analysis"},  # force this tool
            messages=[{"role": "user", "content": f"Analyse this pipeline alert: {alert}"}]
        )

        # Extract the tool call arguments — these are guaranteed to match the schema
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use:
            result = PipelineAlert(**tool_use.input)
            print(f"Alert: {alert[:60]}...")
            print(f"  {result.severity} | {result.category} | flaky={result.is_flaky}")
            print(f"  Fix: {result.suggested_fix[:80]}")
            print()


# ── Technique 5: Batch structured extraction ──────────────────────────────────

def technique_5_batch_extraction():
    """
    Extract structured data from multiple records in one API call.
    More efficient than one call per record.
    """
    print("=== Technique 5: Batch Structured Extraction ===\n")

    sql_queries = [
        "SELECT * FROM orders",
        "SELECT customer_id, SUM(total) FROM orders WHERE date > '2025-01-01' GROUP BY 1 LIMIT 100",
        "DELETE FROM orders WHERE status = 'cancelled'",
        "SELECT o.*, c.name FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.total > 1000",
    ]

    queries_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(sql_queries)])

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0,
        system="""Analyse SQL queries. Return a JSON array, one object per query:
[{"query_type": "SELECT|DML|DDL", "has_where_clause": bool, "has_limit": bool,
  "estimated_complexity": "simple|moderate|complex", "potential_issues": ["list"]}]
Return ONLY the JSON array.""",
        messages=[{"role": "user", "content": f"Analyse these SQL queries:\n{queries_str}"}]
    )

    raw = response.content[0].text.strip()
    try:
        analyses = json.loads(raw)
        print(f"Analysed {len(analyses)} queries:\n")
        for i, (query, analysis) in enumerate(zip(sql_queries, analyses), 1):
            print(f"Query {i}: {query[:50]}...")
            print(f"  Type: {analysis['query_type']} | Complexity: {analysis['estimated_complexity']}")
            if analysis.get("potential_issues"):
                print(f"  Issues: {analysis['potential_issues']}")
            print()
    except json.JSONDecodeError as e:
        print(f"Parse error: {e}\nRaw: {raw[:200]}")


if __name__ == "__main__":
    technique_1_system_prompt()
    technique_2_prefill()
    technique_3_pydantic_retry()
    technique_4_tool_use()
    technique_5_batch_extraction()
