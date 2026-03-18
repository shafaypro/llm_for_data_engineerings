"""
02_tokens_anthropic.py
======================
Concept: Tokens
---------------
Tokens are the unit of text the model reads and writes — not characters,
not words, but subword pieces. "Airflow" might be 1 token; "unfamiliar" might
be 3. You pay per token and your context window is measured in tokens.

Key facts:
- 1 token ≈ 0.75 words ≈ 4 characters (rough rule of thumb)
- 1,000 tokens ≈ 750 words ≈ ~3 pages of text
- Input tokens (what you send) are cheaper than output tokens (what you receive)
- Numbers, code, and special characters are often tokenised inefficiently

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic

client = anthropic.Anthropic()


# ── Demo 1: Count tokens before sending ───────────────────────────────────────
def count_tokens_demo():
    """
    Use the token counting API to estimate cost before making a call.
    Useful for validating that your prompt fits and estimating spend.
    """

    system = "You are a senior data engineer. Be concise."

    messages = [{
        "role": "user",
        "content": """Explain the difference between a fact table and a dimension table
        in a data warehouse. Give a concrete example using an e-commerce schema."""
    }]

    # Count tokens WITHOUT making the actual call
    token_count = client.messages.count_tokens(
        model="claude-haiku-4-5-20251001",
        system=system,
        messages=messages
    )

    print("=== Token Counting Demo ===")
    print(f"Input tokens: {token_count.input_tokens}")
    print(f"Estimated input cost (Haiku ~$0.25/M): ${token_count.input_tokens * 0.00000025:.6f}")

    # Now actually make the call and see real usage
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=system,
        messages=messages
    )

    print(f"\nActual input tokens:  {response.usage.input_tokens}")
    print(f"Actual output tokens: {response.usage.output_tokens}")
    print(f"Total cost estimate:  ${(response.usage.input_tokens * 0.00000025) + (response.usage.output_tokens * 0.00000125):.6f}")
    print(f"\nResponse:\n{response.content[0].text}")


# ── Demo 2: Token efficiency — compare verbose vs concise prompts ──────────────
def token_efficiency_demo():
    """
    Show how prompt wording dramatically affects token usage.
    For pipelines running thousands of times a day, this matters a lot.
    """

    question = "What does GROUP BY do in SQL?"

    # Verbose prompt (common beginner pattern)
    verbose_prompt = f"""Hello! I hope you're doing well today. I have a question that I've been
    wondering about for a while. I'm a data engineer and I work with SQL databases quite a lot
    in my day-to-day work. I was wondering if you could please explain to me, in as much detail
    as you think is necessary, what the GROUP BY clause does in SQL? I'd really appreciate it
    if you could give me a thorough explanation. Thank you so much for your help! Question: {question}"""

    # Tight prompt (production pattern)
    tight_prompt = f"Data engineer question: {question} Answer in 2-3 sentences."

    results = {}
    for name, prompt in [("verbose", verbose_prompt), ("tight", tight_prompt)]:
        count = client.messages.count_tokens(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": prompt}]
        )
        results[name] = count.input_tokens

    print("\n=== Token Efficiency Demo ===")
    print(f"Verbose prompt: {results['verbose']} input tokens")
    print(f"Tight prompt:   {results['tight']} input tokens")
    print(f"Savings:        {results['verbose'] - results['tight']} tokens per call")
    print(f"At 10k calls/day, tight prompt saves ~${(results['verbose'] - results['tight']) * 10000 * 0.00000025:.2f}/day")


# ── Demo 3: Output token control with max_tokens ──────────────────────────────
def max_tokens_demo():
    """
    max_tokens caps how long the response can be.
    Set it tightly for structured outputs (JSON, SQL) — the model
    should be done well before hitting the limit.
    Set it generously for open-ended explanations.
    """

    prompt = "List 10 best practices for writing efficient SQL queries."

    print("\n=== max_tokens Control Demo ===")

    for max_tok in [50, 200, 600]:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tok,
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.content[0].text
        print(f"\nmax_tokens={max_tok} | stop_reason={response.stop_reason} | actual_output_tokens={response.usage.output_tokens}")
        print(output[:200] + ("..." if len(output) > 200 else ""))


if __name__ == "__main__":
    count_tokens_demo()
    token_efficiency_demo()
    max_tokens_demo()
