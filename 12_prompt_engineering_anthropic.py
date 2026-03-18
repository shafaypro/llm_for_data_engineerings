"""
12_prompt_engineering_anthropic.py
===================================
Concept: Prompt Engineering
----------------------------
Prompt engineering is the craft of writing instructions that reliably
produce the output you want. Like query optimisation — the same logical
request written differently produces vastly different results, and there's
no compiler to catch mistakes. You iterate empirically.

Key techniques covered here:
1. Be explicit about format (JSON schema, column names, no preamble)
2. Few-shot examples — show 2-3 ideal input/output pairs
3. Chain of thought — "think step by step" improves reasoning
4. XML tags — Claude responds well to structured prompts with tags
5. Negative examples — show what NOT to do
6. Role + context — who you are and what you know
7. Output length control

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json

client = anthropic.Anthropic()


# ── Technique 1: Be explicit about format ─────────────────────────────────────
def explicit_format_demo():
    """
    Vague prompts get vague outputs. Explicit format contracts get structured outputs.
    Show the exact shape you want — including field names, types, and constraints.
    """

    print("=== Technique 1: Explicit Format ===\n")

    question = "Analyse the health of our orders pipeline"

    # Vague
    vague_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": question}]
    )

    # Explicit
    explicit_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        temperature=0,
        system="""Respond ONLY with valid JSON — no preamble, no explanation, no markdown code blocks.
Use exactly this schema:
{
  "status": "healthy | degraded | critical",
  "score": 0-100,
  "issues": ["string"],
  "action_required": true | false
}""",
        messages=[{"role": "user", "content": question}]
    )

    print("Vague prompt output:")
    print(vague_response.content[0].text[:200])
    print("\nExplicit format output:")
    print(explicit_response.content[0].text)
    print()


# ── Technique 2: Few-shot examples ────────────────────────────────────────────
def few_shot_demo():
    """
    Show the model exactly what good output looks like with 2-3 examples.
    This is the most powerful technique for consistent structured output.
    The examples go in the system prompt or as alternating user/assistant turns.
    """

    print("=== Technique 2: Few-Shot Examples ===\n")

    # Without examples
    no_examples = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        temperature=0,
        system="Convert natural language requests to dbt model names using our naming conventions.",
        messages=[{"role": "user", "content": "daily revenue by product category"}]
    )

    # With few-shot examples in the conversation history (most reliable method)
    with_examples = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=32,
        temperature=0,
        system="""You convert natural language descriptions to dbt model names.
Naming rules: stg_ for staging, fct_ for facts, dim_ for dimensions, int_ for intermediate.
Double underscore separates source from entity: stg_stripe__payments.
Respond with ONLY the model name, nothing else.""",
        messages=[
            {"role": "user", "content": "cleans raw stripe payment records"},
            {"role": "assistant", "content": "stg_stripe__payments"},
            {"role": "user", "content": "one row per order with revenue metrics"},
            {"role": "assistant", "content": "fct_orders"},
            {"role": "user", "content": "all customer attributes and segments"},
            {"role": "assistant", "content": "dim_customers"},
            # The actual request:
            {"role": "user", "content": "daily revenue by product category"},
        ]
    )

    print(f"Without examples: {no_examples.content[0].text.strip()[:100]}")
    print(f"With examples:    {with_examples.content[0].text.strip()}")
    print()


# ── Technique 3: Chain of thought ─────────────────────────────────────────────
def chain_of_thought_demo():
    """
    For complex reasoning tasks (debugging, root cause analysis, multi-step logic),
    asking the model to "think step by step" before answering improves accuracy.
    The model uses the scratchpad to work through the problem.
    """

    print("=== Technique 3: Chain of Thought ===\n")

    problem = """Our daily revenue dropped 35% yesterday compared to the day before.
    Here's what we know:
    - Total orders: same count (1,200 orders both days)
    - New customer signups: normal
    - Stripe payments API: no incidents reported
    - We deployed a promo code feature yesterday afternoon
    - Average order value: dropped from $118 to $77
    What is likely the root cause?"""

    # Without CoT
    direct = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": problem}]
    )

    # With CoT
    cot = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": f"""{problem}

Think through this step by step before giving your answer:
1. What does the data tell us?
2. What can we rule out?
3. What is the most likely explanation?
Then give your final answer."""}]
    )

    print("Without chain of thought:")
    print(direct.content[0].text.strip()[:200])
    print("\nWith chain of thought:")
    print(cot.content[0].text.strip()[:500])
    print()


# ── Technique 4: XML tags for structure ───────────────────────────────────────
def xml_tags_demo():
    """
    Claude is trained to pay close attention to XML-style tags.
    Use them to clearly delineate different parts of your prompt:
    instructions, context, data, examples, constraints.
    This prevents the model from confusing instructions with data.
    """

    print("=== Technique 4: XML Tags ===\n")

    # Imagine this comes from your pipeline at runtime
    sql_query = "SELECT * FROM orders WHERE customer_id = '{{customer_id}}'"
    schema_info = "Table: orders(order_id UUID, customer_id UUID, total DECIMAL, created_at TIMESTAMPTZ)"
    user_question = "Is this query safe to run in production? What could go wrong?"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="You are a data engineering code reviewer. Be concise and specific.",
        messages=[{"role": "user", "content": f"""<context>
<schema>{schema_info}</schema>
<query>{sql_query}</query>
</context>

<instructions>
Review the query above and answer the question below.
Focus on: security (SQL injection), performance (indexes, full scans), correctness.
</instructions>

<question>{user_question}</question>

<output_format>
Respond with a JSON object:
{{"safe": true/false, "issues": ["list of issues"], "recommendation": "fixed query or explanation"}}
</output_format>"""}]
    )

    print(response.content[0].text)
    print()


# ── Technique 5: Negative examples ────────────────────────────────────────────
def negative_examples_demo():
    """
    Telling the model what NOT to do is sometimes more effective than
    telling it what to do. Show a bad example explicitly.
    """

    print("=== Technique 5: Negative Examples ===\n")

    system = """You write dbt test YAML for data quality checks.

BAD output (never do this):
```yaml
# Lots of markdown
Here's the test:
version: 2
```
The above is wrong because it includes markdown, code fences, and a comment.

GOOD output (always do this — raw YAML only, no other text):
version: 2
models:
  - name: fct_orders
    columns:
      - name: order_id
        tests:
          - not_null
          - unique

Always output raw YAML only. Never include markdown fences, comments, or explanation."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": "Write dbt tests for: fct_revenue(revenue_date DATE, region VARCHAR, net_revenue DECIMAL)"}]
    )

    print("Output (should be raw YAML, no markdown):")
    print(response.content[0].text)
    print()


# ── Technique 6: Output length control ────────────────────────────────────────
def length_control_demo():
    """
    Explicitly specify how long the response should be.
    Vague prompts often get verbose outputs. Be direct about length.
    """

    print("=== Technique 6: Output Length Control ===\n")

    topic = "How does Snowflake's micro-partition pruning work?"

    lengths = [
        ("one sentence", 50),
        ("3 bullet points", 150),
        ("a detailed explanation with examples", 500),
    ]

    for length_instruction, max_tok in lengths:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tok,
            messages=[{"role": "user", "content": f"Explain in {length_instruction}: {topic}"}]
        )
        output = response.content[0].text.strip()
        words = len(output.split())
        print(f"Instruction: '{length_instruction}' ({words} words):")
        print(output[:300])
        print()


# ── Prompt quality checklist ──────────────────────────────────────────────────
def prompt_checklist():
    """Print a quick-reference checklist for writing production prompts."""

    print("=== Prompt Engineering Checklist ===\n")
    checks = [
        ("Format", "Specify exact output format (JSON schema, column names, no preamble)"),
        ("Temperature", "Use 0 for structured output, 0.3-0.7 for prose"),
        ("Examples", "Include 2-3 few-shot examples for structured tasks"),
        ("XML tags", "Use <context>, <instructions>, <output_format> to separate concerns"),
        ("Negative", "Add 'never do X' examples for common failure modes"),
        ("Length", "State explicitly: '1 sentence', '3 bullets', 'under 100 words'"),
        ("Role", "Set persona: 'You are a senior data engineer at a fintech company'"),
        ("Constraints", "List hard rules: 'Only SELECT statements', 'Always include LIMIT'"),
        ("Validation", "Parse and validate the output before using it in production"),
        ("Eval", "Write 3-5 regression tests that run on every prompt change"),
    ]

    for name, desc in checks:
        print(f"  ✓ {name:<14} {desc}")


if __name__ == "__main__":
    explicit_format_demo()
    few_shot_demo()
    chain_of_thought_demo()
    xml_tags_demo()
    negative_examples_demo()
    length_control_demo()
    prompt_checklist()
