# LLM Concepts — Code Examples for Data Engineers
# =================================================
# All examples use the Anthropic Python SDK.
# Every file is self-contained and runnable.
#
# SETUP
# -----
# pip install anthropic voyageai numpy pyyaml
# export ANTHROPIC_API_KEY=sk-ant-...
# export VOYAGE_API_KEY=pa-...   (only needed for file 03 and 06)
#
# FILES
# -----
# 01_context_window_anthropic.py   — What fits in one API call, how to manage large inputs
# 02_tokens_anthropic.py           — Count tokens, estimate cost, control output length
# 03_embeddings_anthropic.py       — Semantic similarity, in-memory vector store, mini RAG
# 04_temperature_anthropic.py      — Deterministic vs creative output, task-by-task guide
# 05_system_prompt_anthropic.py    — Role framing, output contracts, constraints, few-shot
# 06_rag_anthropic.py              — Full RAG pipeline: chunk → embed → store → retrieve → answer
# 07_tool_use_anthropic.py         — Agent loop: Claude calls run_sql, get_dag_status, etc.
# 08_streaming_anthropic.py        — Stream tokens live, latency comparison, Slack bot pattern
# 09_hallucination_and_evals_anthropic.py — Mitigations, output validation, LLM-as-judge, test suite
# 10_cost_and_caching_anthropic.py — Prompt caching, Batch API, cost calculator
# 11_multimodal_anthropic.py       — Images, PDFs, ERD screenshots, dashboard comparison
# 12_prompt_engineering_anthropic.py — Explicit format, few-shot, CoT, XML tags, length control
#
# RECOMMENDED LEARNING ORDER
# --------------------------
# Day 1:   01 → 02 → 04 → 05   (fundamentals, get comfortable with the API)
# Day 2:   08 → 09 → 12        (production patterns, prompt craft)
# Week 2:  03 → 06             (embeddings and RAG — needs voyageai key)
# Week 2:  07                   (tool use / agents — the fun part)
# Week 3:  10 → 11             (cost optimisation, multimodal)

## Quick start (< 5 minutes)

# 1. Install
#    pip install anthropic

# 2. Get API key
#    https://console.anthropic.com → API Keys → Create Key

# 3. Set env var
#    export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run your first example
#    python 01_context_window_anthropic.py
