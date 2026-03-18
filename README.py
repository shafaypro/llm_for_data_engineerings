# 🔧 LLM for Data Engineers

> **Practical LLM engineering for data engineers — every concept explained with runnable Python code, mapped to the tools you already know: Airflow, dbt, Snowflake, AWS, and GCP.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange.svg)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 🧭 Who This Is For

You are a **data engineer** who knows:

- ✅ Airflow, dbt, Snowflake / BigQuery
- ✅ Python, SQL, AWS / GCP
- ✅ Terraform, Spark, Kafka
- ✅ Maybe some data science, Tableau, QuickSight

But you want to add **AI agent building** to your skillset without sitting through generic LLM tutorials that assume you're a software engineer with no infra background.

This repo is written **by a data engineer, for data engineers**. Every concept maps to something you already know. Every code file is self-contained, runnable, and directly applicable to real data pipelines.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/llm-for-data-engineers.git
cd llm-for-data-engineers

# 2. Install
pip install anthropic voyageai numpy pydantic openai

# 3. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...   # get from console.anthropic.com

# 4. Run your first file
python fundamentals/01_context_window_anthropic.py
```

You'll have a working LLM call in under 5 minutes.

---

## 📁 Repository Structure

```
llm-for-data-engineers/
├── fundamentals/          # Core LLM concepts (files 01–12)
├── advanced/              # Production patterns (files 13–19)
├── projects/              # Real end-to-end projects
├── requirements.txt
└── README.md
```

---

## 📚 Fundamentals (Files 01–12)

> Start here. These 12 files cover every core LLM concept with a **data engineering analogy** so nothing feels foreign.

| # | File | Concept | Data Engineering Analogy |
|---|------|---------|--------------------------|
| 01 | `01_context_window_anthropic.py` | Context window | Working memory of a single SQL query execution |
| 02 | `02_tokens_anthropic.py` | Tokens | Bytes in a file — not characters, not words |
| 03 | `03_embeddings_anthropic.py` | Embeddings | Feature vectors — numeric representation of text |
| 04 | `04_temperature_anthropic.py` | Temperature | Adding noise/jitter to pipeline output |
| 05 | `05_system_prompt_anthropic.py` | System prompt | Config file / environment variables |
| 06 | `06_rag_anthropic.py` | RAG | Parameterised query that fetches rows then calls a function |
| 07 | `07_tool_use_anthropic.py` | Tool use / agents | Airflow task that decides its own branch at runtime |
| 08 | `08_streaming_anthropic.py` | Streaming | Kafka consumer vs waiting for a full batch file |
| 09 | `09_hallucination_and_evals_anthropic.py` | Hallucination + evals | Silent data corruption — like dbt tests catching it |
| 10 | `10_cost_and_caching_anthropic.py` | Cost + prompt caching | Cloud compute billing — pay per unit of work |
| 11 | `11_multimodal_anthropic.py` | Multimodal inputs | Pipeline accepting CSV, JSON, Parquet, PDF |
| 12 | `12_prompt_engineering_anthropic.py` | Prompt engineering | Query optimisation — same logic, better phrasing |

### What's inside each file

Every fundamentals file contains **3–5 runnable demos** that progressively build on each other. For example `06_rag_anthropic.py` walks you through:

1. Chunking documents into overlapping pieces
2. Embedding each chunk with Voyage AI
3. Building an in-memory vector store with cosine similarity search
4. Retrieving the top-k relevant chunks for a query
5. Calling Claude with the retrieved context to answer grounded questions

---

## 🔬 Advanced (Files 13–19)

> Once you're comfortable with the fundamentals, these 7 files cover production-grade patterns used in real AI-powered data platforms.

| # | File | Concept | Key Patterns |
|---|------|---------|--------------|
| 13 | `13_multi_agent_orchestration_anthropic.py` | Multi-agent systems | Orchestrator/specialists, sequential pipelines, parallel fan-out, critic loop |
| 14 | `14_structured_outputs_anthropic.py` | Structured outputs | Pydantic validation, retry loop, tool-use as schema enforcement |
| 15 | `15_memory_and_state_anthropic.py` | Memory & state | Sliding window, summarisation, external persistent memory |
| 16 | `16_advanced_rag_anthropic.py` | Advanced RAG | Hybrid search, re-ranking, HyDE, query decomposition, metadata filtering |
| 17 | `17_observability_anthropic.py` | LLM observability | Latency, cost, quality scoring, prompt A/B tracking, regression detection |
| 18 | `18_local_models_ollama.py` | Local models | Ollama, quantization, zero-cost dev, drop-in cloud replacement |
| 19 | `19_llm_data_pipeline_patterns_anthropic.py` | Pipeline patterns | Schema inference, DQ triage, NL-to-SQL, anomaly explanation, self-healing |

### Highlight: File 19 — LLM Data Pipeline Patterns

This is the most directly applicable file for working data engineers. It contains 6 production patterns you can drop into existing pipelines today:

```python
# Pattern 6: Self-healing pipeline — wire this into your Airflow on_failure_callback
fix = suggest_pipeline_fix(
    dag_id="daily_orders_etl",
    task_id="load_to_snowflake",
    error_log=error_log,
    historical_fixes=past_fixes  # learns from your team's history
)
# Returns: root cause, ranked fix steps, config changes, retry safety, ETA
```

---

## 🏗️ Projects

Four real projects that go from zero to production in a weekend:

### Project 1 — Airflow Log Summariser (~15 lines)
Reads a failed DAG run log, calls Claude, returns structured JSON with `error_type`, `root_cause`, `suggested_fix`, and `severity`. Your hello world.

```bash
python projects/summarise_log.py my_dag.log
```

### Project 2 — SQL Query Explainer (~40 lines)
Interactive session: paste a complex query, get a plain-English explanation + optimisation suggestions. Supports multi-turn follow-up questions.

```bash
python projects/sql_explainer.py
```

### Project 3 — dbt Doc Generator (~55 lines)
Reads your `schema.yml`, finds models with missing descriptions, generates them with Claude, writes them back. Run it on your actual dbt project.

```bash
python projects/dbt_doc_generator.py models/schema.yml
```

### Project 4 — Airflow Failure Bot (~60 lines)
`on_failure_callback` that grabs the task log, calls Claude, posts a rich colour-coded Slack message with root cause + suggested fix. Wire it into any DAG with one line.

```python
# In your DAG file:
from projects.airflow_failure_callback import on_failure_callback

with DAG("my_dag", default_args={"on_failure_callback": on_failure_callback}):
    ...
```

---

## 🧠 Concept Map

How everything connects:

```
FUNDAMENTALS                    ADVANCED
─────────────────────────────────────────────────────────
Context Window ──────────────► Memory & State (15)
Tokens + Cost ───────────────► Observability (17)
Embeddings ──────────────────► Advanced RAG (16)
RAG ─────────────────────────► Advanced RAG (16)
Tool Use ────────────────────► Multi-Agent (13)
Temperature + Prompting ─────► Structured Outputs (14)
Streaming ───────────────────► Pipeline Patterns (19)
Evals ───────────────────────► Observability (17)
Local Models ────────────────► Pipeline Patterns (19)
```

---

## 💡 Key Concepts vs Your Existing Stack

| LLM Concept | Your DE Equivalent |
|-------------|-------------------|
| Context window | Working memory of a SQL query |
| Tokens | Bytes — you pay per unit |
| Embeddings | Feature vectors |
| Temperature | Controlled randomness / noise |
| System prompt | Environment config / `airflow.cfg` |
| RAG | Runtime data injection into context |
| Tool use | Airflow branch operator with reasoning |
| Fine-tuning | Retraining a model on your own labels |
| Hallucination | Silent data corruption — no error thrown |
| Prompt caching | Query result caching — 90% cheaper on repeats |
| Quantization | Parquet compression — smaller file, tiny quality loss |

---

## ⚙️ Requirements

```txt
anthropic>=0.40.0
voyageai>=0.3.0        # embeddings (files 03, 06, 16)
numpy>=1.26.0
pydantic>=2.0.0        # structured outputs (file 14)
openai>=1.0.0          # Ollama compatibility (file 18)
pyyaml>=6.0.0          # dbt doc generator project
```

Install everything:
```bash
pip install -r requirements.txt
```

API keys needed:
- `ANTHROPIC_API_KEY` — from [console.anthropic.com](https://console.anthropic.com)
- `VOYAGE_API_KEY` — from [voyageai.com](https://voyageai.com) (free tier available, files 03/06/16 only)

---

## 📅 Suggested Learning Schedule

| Timeline | Files | Focus |
|----------|-------|-------|
| Day 1 | 01, 02, 04, 05 | Get comfortable with the API, prompting basics |
| Day 2–3 | 08, 09, 12 | Streaming, evals, prompt engineering for production |
| Week 2 | 03, 06 | Embeddings and RAG — needs Voyage key |
| Week 2 | 07 | Tool use and agents — the fun part |
| Week 3 | 10, 11 | Cost optimisation, multimodal inputs |
| Week 3–4 | 13, 14, 15 | Multi-agent, structured outputs, memory |
| Week 4+ | 16, 17, 18, 19 | Advanced RAG, observability, local models, pipeline patterns |

---

## 🗺️ Roadmap (Upcoming Files)

- [ ] `20_fine_tuning_anthropic.py` — when and how to fine-tune vs RAG
- [ ] `21_langgraph_orchestration.py` — LangGraph for complex agent workflows
- [ ] `22_airflow_ai_operators.py` — custom Airflow operators that call LLMs
- [ ] `23_dbt_ai_copilot.py` — full dbt model writer + tester agent
- [ ] `24_snowflake_cortex.py` — native LLM functions inside Snowflake SQL
- [ ] `25_vector_db_pgvector.py` — production RAG with pgvector on Postgres

---

## 🤝 Contributing

This is an ongoing learning log. If you're a data engineer building with LLMs and want to add a pattern, fix a bug, or improve an explanation:

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-pattern-name`
3. Follow the file naming convention: `XX_concept_name_anthropic.py`
4. Every file must include: concept explanation, DE analogy, 3+ runnable demos
5. Open a PR with a description of what the file teaches

---

## 📖 Further Reading

- [Anthropic API Docs](https://docs.anthropic.com) — official reference
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Voyage AI Embeddings](https://docs.voyageai.com) — embedding models used in this repo
- [Ollama](https://ollama.com) — local model runner (file 18)
- [dbt docs](https://docs.getdbt.com) — for the dbt integration patterns

---

## 📜 License

MIT — use freely, attribution appreciated.

---

<div align="center">
  <strong>Built by a data engineer who wanted better AI resources for the DE community.</strong>
  <br>
  If this helped you, ⭐ the repo and share it with your data team.
</div>
