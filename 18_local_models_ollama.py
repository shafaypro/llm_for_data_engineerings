"""
18_local_models_ollama.py
==========================
Concept: Running Models Locally with Ollama
--------------------------------------------
Ollama runs open-source LLMs locally on your machine.
No API costs, no data leaving your network, no rate limits.
The API is OpenAI-compatible so you can swap between local and cloud
with one line change.

Use cases for local models in data engineering:
- Development and testing (no cost)
- Processing sensitive data (no external API calls)
- High-volume classification jobs (no per-token billing)
- Air-gapped environments

Model sizes vs hardware requirements:
- llama3.2:1b  — 1.3GB  — runs on any laptop (8GB RAM)
- llama3.2:3b  — 2.0GB  — good quality on laptop
- llama3.1:8b  — 4.7GB  — strong quality, needs 8GB VRAM or 16GB RAM
- llama3.1:70b — 40GB   — needs 64GB RAM or multi-GPU
- mistral:7b   — 4.1GB  — great for code and structured output
- codellama:7b — 3.8GB  — optimised for code tasks

Install Ollama: https://ollama.com (brew install ollama on Mac)
Then: ollama pull llama3.2:3b
pip install anthropic openai   (Ollama has OpenAI-compatible API)
"""

import json
import time
import subprocess


# ── Check if Ollama is running ────────────────────────────────────────────────

def check_ollama() -> bool:
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


# ── Option A: Use via OpenAI-compatible client ────────────────────────────────

def demo_openai_compatible():
    """
    Ollama exposes an OpenAI-compatible API.
    This means you can use the openai Python library to talk to local models.
    Swap base_url to switch between local and cloud.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("Install with: pip install openai")
        return

    # Local Ollama — same code, different base_url
    local_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # required by library but ignored by Ollama
    )

    print("=== Ollama via OpenAI-compatible API ===\n")

    if not check_ollama():
        print("Ollama not running. Start with: ollama serve")
        print("Then pull a model: ollama pull llama3.2:3b")
        return

    prompt = "Classify this Airflow alert as P1/P2/P3. Return JSON only: {'severity': 'P1|P2|P3'}. Alert: Snowflake warehouse auto-suspended."

    start = time.time()
    response = local_client.chat.completions.create(
        model="llama3.2:3b",   # or any model you've pulled
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    latency = (time.time() - start) * 1000

    print(f"Model: llama3.2:3b (local)")
    print(f"Output: {response.choices[0].message.content}")
    print(f"Latency: {latency:.0f}ms")
    print(f"Cost: $0.00\n")


# ── Option B: Use Ollama's native Python library ──────────────────────────────

def demo_ollama_native():
    """
    The ollama Python library is the cleanest way to use local models.
    pip install ollama
    """
    try:
        import ollama
    except ImportError:
        print("Install with: pip install ollama")
        return

    if not check_ollama():
        print("Ollama not running.")
        return

    print("=== Ollama Native Library ===\n")

    # Basic call
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "What is a dbt model in 2 sentences?"}]
    )
    print(f"Response: {response['message']['content'][:200]}")

    # Streaming
    print("\nStreaming response: ", end="", flush=True)
    for chunk in ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "Name 3 Airflow best practices, very briefly"}],
        stream=True
    ):
        print(chunk['message']['content'], end="", flush=True)
    print()


# ── Option C: Drop-in replacement for Anthropic calls ────────────────────────

def demo_drop_in_replacement():
    """
    The cleanest pattern: write one function that can use either
    Claude (via Anthropic API) or a local model (via Ollama).
    Switch with an environment variable for dev/prod separation.
    """
    import os

    def call_llm(prompt: str, system: str = "", use_local: bool = False) -> str:
        """
        Universal LLM caller.
        use_local=True  → Ollama (free, private, slower)
        use_local=False → Anthropic Claude (paid, fast, high quality)
        """
        if use_local:
            try:
                from openai import OpenAI
                c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                resp = c.chat.completions.create(
                    model=os.getenv("LOCAL_MODEL", "llama3.2:3b"),
                    messages=messages,
                    temperature=0
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f"[Local model error: {e}]"
        else:
            import anthropic
            c = anthropic.Anthropic()
            kwargs = {"model": "claude-haiku-4-5-20251001", "max_tokens": 256, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]}
            if system:
                kwargs["system"] = system
            resp = c.messages.create(**kwargs)
            return resp.content[0].text.strip()

    print("=== Drop-in Replacement Pattern ===\n")
    print("Switching between local and cloud with one flag:\n")

    system = 'Classify alerts. Return JSON only: {"severity": "P1|P2|P3"}'
    alert = "Airflow scheduler has not sent a heartbeat in 10 minutes"

    for use_local in [False, True]:
        mode = "LOCAL (Ollama)" if use_local else "CLOUD (Claude)"
        if use_local and not check_ollama():
            print(f"{mode}: Ollama not running — skipping")
            continue
        start = time.time()
        result = call_llm(alert, system=system, use_local=use_local)
        latency = (time.time() - start) * 1000
        print(f"{mode} ({latency:.0f}ms):")
        print(f"  {result[:120]}\n")


# ── Model capability comparison ───────────────────────────────────────────────

def demo_model_comparison():
    """
    Compare output quality between a small local model and Claude.
    Run this to calibrate which tasks need cloud vs local.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return

    import anthropic

    if not check_ollama():
        print("Ollama not running — skipping comparison")
        return

    print("=== Model Comparison: Local vs Cloud ===\n")

    local = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    cloud = anthropic.Anthropic()

    tasks = [
        {
            "name": "Simple classification",
            "prompt": "Classify as P1/P2/P3. Return JSON only. Alert: S3 bucket access denied.",
            "system": '{"severity": "P1|P2|P3"}'
        },
        {
            "name": "SQL generation",
            "prompt": "Write SQL to get top 5 customers by revenue last 30 days. Tables: orders(id,customer_id,created_at,amount), customers(id,name). Return SQL only.",
            "system": None
        },
        {
            "name": "Complex reasoning",
            "prompt": "Revenue dropped 40% but order count is the same. AOV dropped. We deployed a discount feature yesterday. What is likely the root cause? Be specific.",
            "system": None
        }
    ]

    for task in tasks:
        print(f"Task: {task['name']}")

        # Local
        messages = []
        if task["system"]:
            messages.append({"role": "system", "content": task["system"]})
        messages.append({"role": "user", "content": task["prompt"]})

        t = time.time()
        local_resp = local.chat.completions.create(
            model="llama3.2:3b", messages=messages, temperature=0
        )
        local_time = (time.time() - t) * 1000
        print(f"  Local (llama3.2:3b, {local_time:.0f}ms): {local_resp.choices[0].message.content[:120]}")

        # Cloud
        t = time.time()
        kwargs = {"model": "claude-haiku-4-5-20251001", "max_tokens": 200, "temperature": 0,
                  "messages": [{"role": "user", "content": task["prompt"]}]}
        if task["system"]:
            kwargs["system"] = task["system"]
        cloud_resp = cloud.messages.create(**kwargs)
        cloud_time = (time.time() - t) * 1000
        print(f"  Cloud (haiku, {cloud_time:.0f}ms):     {cloud_resp.content[0].text.strip()[:120]}")
        print()


# ── Quantization reference ────────────────────────────────────────────────────

def print_quantization_guide():
    print("=== Ollama Model + Quantization Quick Reference ===\n")
    models = [
        ("llama3.2:1b",   "1.3GB", "M1 Mac / any laptop",     "Basic tasks, very fast"),
        ("llama3.2:3b",   "2.0GB", "Any modern laptop",        "Good for classification, SQL"),
        ("llama3.1:8b",   "4.7GB", "16GB RAM laptop",          "Strong reasoning, code"),
        ("mistral:7b",    "4.1GB", "16GB RAM laptop",          "Excellent structured output"),
        ("codellama:7b",  "3.8GB", "16GB RAM laptop",          "Optimised for code/SQL"),
        ("llama3.1:70b",  "40GB",  "64GB RAM or multi-GPU",    "Near-GPT4 quality"),
        ("llama3.1:70b-q4_0", "35GB", "64GB RAM",              "70B quantised to int4"),
    ]

    print(f"{'Model':<25} {'Size':>6}  {'Hardware':<25} {'Best for'}")
    print("-" * 80)
    for model, size, hw, use in models:
        print(f"{model:<25} {size:>6}  {hw:<25} {use}")

    print("\nQuick start:")
    print("  brew install ollama          # Mac")
    print("  ollama serve                 # start server")
    print("  ollama pull llama3.2:3b      # download model")
    print("  ollama run llama3.2:3b       # interactive chat")
    print("  # API at http://localhost:11434")


if __name__ == "__main__":
    print_quantization_guide()
    print()
    demo_openai_compatible()
    demo_ollama_native()
    demo_drop_in_replacement()
    demo_model_comparison()
