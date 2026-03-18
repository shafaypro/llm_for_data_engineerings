"""
ADVANCED LLM CONCEPTS — Code Examples for Data Engineers
=========================================================
Second pack — builds on the 12-file fundamentals pack.

SETUP
-----
pip install anthropic voyageai numpy pydantic openai ollama
export ANTHROPIC_API_KEY=sk-ant-...
export VOYAGE_API_KEY=pa-...   (files 16 only)

FILES
-----
13_multi_agent_orchestration_anthropic.py
    Orchestrator + specialists, sequential pipelines, parallel fan-out,
    critic/self-refinement loops. The architecture behind production AI systems.

14_structured_outputs_anthropic.py
    5 techniques for reliable JSON: system prompt contracts, prefill,
    Pydantic + retry loop, tool-use as schema enforcement, batch extraction.

15_memory_and_state_anthropic.py
    LLMs have no memory. 4 strategies: full history, sliding window,
    summarisation, external persistent memory across sessions.

16_advanced_rag_anthropic.py
    Beyond basic RAG: hybrid search, re-ranking, query decomposition,
    HyDE (Hypothetical Document Embedding), metadata filtering.

17_observability_anthropic.py
    Production monitoring: latency, cost, quality scoring, prompt version
    A/B tracking, regression detection. The dbt-tests equivalent for LLMs.

18_local_models_ollama.py
    Run models locally with Ollama. Zero cost, private data, no rate limits.
    Drop-in replacement pattern for dev vs prod switching.

19_llm_data_pipeline_patterns_anthropic.py
    Real production patterns: schema inference, DQ triage, NL-to-SQL,
    lineage docs, anomaly explanation, self-healing pipelines.

LEARNING ORDER
--------------
Week 2-3:  13 → 14 → 15   (agents, structured output, memory)
Week 3-4:  16 → 17        (advanced RAG, observability)
Week 4+:   18 → 19        (local models, pipeline integration)

WHAT TO BUILD AFTER THESE
--------------------------
With all 19 files under your belt, you're ready for:
- A full AI-powered data platform assistant (combines 13+15+16+19)
- A self-monitoring pipeline with auto-triage (combines 17+19)
- A private local model deployment for sensitive data (18)
- A dbt co-pilot that writes, documents, and tests models (14+19)
"""
