"""
17_observability_anthropic.py
==============================
Concept: LLM Observability & Monitoring
-----------------------------------------
Running LLMs in production without observability is like running a data
pipeline without logging or alerting. You need to track: latency, cost,
error rates, output quality, and prompt regressions.

This file shows how to build a lightweight observability layer you can
drop into any existing LLM pipeline.

Tracks:
- Latency (TTFT, total time, tokens/sec)
- Cost per call and cumulative
- Error rates and failure modes
- Output quality scores
- Prompt version tracking
- Slow / expensive call detection

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from functools import wraps

client = anthropic.Anthropic()

# Cost per token (update from docs.anthropic.com)
COST_PER_TOKEN = {
    "claude-haiku-4-5-20251001":  {"input": 0.25e-6,  "output": 1.25e-6},
    "claude-sonnet-4-6": {"input": 3.00e-6,  "output": 15.00e-6},
}


# ── Observation data model ────────────────────────────────────────────────────

@dataclass
class LLMObservation:
    call_id: str
    timestamp: str
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: Optional[float]
    cost_usd: float
    success: bool
    error: Optional[str] = None
    quality_score: Optional[float] = None
    tags: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0:
            return 0
        return self.output_tokens / (self.latency_ms / 1000)


class ObservabilityStore:
    """In-memory store. In production: write to Postgres, BigQuery, or Datadog."""

    def __init__(self, log_file: Optional[str] = None):
        self.observations: List[LLMObservation] = []
        self.log_file = Path(log_file) if log_file else None

    def record(self, obs: LLMObservation) -> None:
        self.observations.append(obs)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(asdict(obs)) + "\n")

        # Alert on slow or expensive calls
        if obs.latency_ms > 10000:
            print(f"  ⚠ SLOW CALL: {obs.latency_ms:.0f}ms (call_id={obs.call_id})")
        if obs.cost_usd > 0.01:
            print(f"  ⚠ EXPENSIVE CALL: ${obs.cost_usd:.4f} (call_id={obs.call_id})")
        if not obs.success:
            print(f"  ✗ FAILED CALL: {obs.error} (call_id={obs.call_id})")

    def summary(self) -> dict:
        if not self.observations:
            return {}
        successful = [o for o in self.observations if o.success]
        failed = [o for o in self.observations if not o.success]
        return {
            "total_calls": len(self.observations),
            "success_rate": len(successful) / len(self.observations),
            "total_cost_usd": sum(o.cost_usd for o in self.observations),
            "avg_latency_ms": sum(o.latency_ms for o in successful) / max(len(successful), 1),
            "avg_tokens_per_sec": sum(o.tokens_per_second for o in successful) / max(len(successful), 1),
            "total_input_tokens": sum(o.input_tokens for o in self.observations),
            "total_output_tokens": sum(o.output_tokens for o in self.observations),
            "error_count": len(failed),
            "errors": list({o.error for o in failed if o.error}),
        }

    def by_prompt_version(self) -> dict:
        versions = {}
        for obs in self.observations:
            v = obs.prompt_version
            if v not in versions:
                versions[v] = {"calls": 0, "total_cost": 0, "avg_latency": [], "quality_scores": []}
            versions[v]["calls"] += 1
            versions[v]["total_cost"] += obs.cost_usd
            versions[v]["avg_latency"].append(obs.latency_ms)
            if obs.quality_score is not None:
                versions[v]["quality_scores"].append(obs.quality_score)

        return {
            v: {
                "calls": d["calls"],
                "total_cost": d["total_cost"],
                "avg_latency_ms": sum(d["avg_latency"]) / len(d["avg_latency"]) if d["avg_latency"] else 0,
                "avg_quality": sum(d["quality_scores"]) / len(d["quality_scores"]) if d["quality_scores"] else None,
            }
            for v, d in versions.items()
        }


# ── Observed client wrapper ────────────────────────────────────────────────────

class ObservedAnthropicClient:
    """
    Wraps the Anthropic client to automatically log every call.
    Drop-in replacement: just change `client = anthropic.Anthropic()`
    to `client = ObservedAnthropicClient(store, prompt_version="v1.2")`
    """

    def __init__(self, store: ObservabilityStore, prompt_version: str = "v1.0"):
        self._client = anthropic.Anthropic()
        self.store = store
        self.prompt_version = prompt_version
        self.messages = self  # mimic client.messages interface

    def create(self, **kwargs) -> any:
        call_id = str(uuid.uuid4())[:8]
        model = kwargs.get("model", "unknown")
        start = time.time()
        ttft = None
        error = None
        response = None

        try:
            response = self._client.messages.create(**kwargs)
            latency_ms = (time.time() - start) * 1000

            prices = COST_PER_TOKEN.get(model, COST_PER_TOKEN["claude-haiku-4-5-20251001"])
            cost = (response.usage.input_tokens * prices["input"] +
                    response.usage.output_tokens * prices["output"])

            obs = LLMObservation(
                call_id=call_id,
                timestamp=datetime.utcnow().isoformat(),
                model=model,
                prompt_version=self.prompt_version,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                ttft_ms=ttft,
                cost_usd=cost,
                success=True,
                tags={"has_system": "system" in kwargs, "has_tools": "tools" in kwargs}
            )
            self.store.record(obs)
            return response

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            error = str(e)
            obs = LLMObservation(
                call_id=call_id,
                timestamp=datetime.utcnow().isoformat(),
                model=model,
                prompt_version=self.prompt_version,
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                ttft_ms=None,
                cost_usd=0,
                success=False,
                error=error
            )
            self.store.record(obs)
            raise


# ── Demo 1: Basic observability ───────────────────────────────────────────────

def basic_observability_demo():
    print("=== Basic Observability Demo ===\n")

    store = ObservabilityStore(log_file="/tmp/llm_calls.jsonl")
    observed_client = ObservedAnthropicClient(store, prompt_version="v1.0")

    tasks = [
        "Classify: 'Snowflake warehouse suspended'",
        "Classify: 'dbt model exceeded SLA'",
        "Classify: 'Fivetran sync failed auth'",
        "What is a window function in SQL? Explain in detail with examples.",
    ]

    for task in tasks:
        response = observed_client.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            temperature=0,
            messages=[{"role": "user", "content": task}]
        )
        print(f"Task: {task[:50]}...")
        print(f"  Output: {response.content[0].text[:80]}...")

    print("\n--- Call Summary ---")
    summary = store.summary()
    for k, v in summary.items():
        if k != "errors":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


# ── Demo 2: Prompt version A/B tracking ──────────────────────────────────────

def prompt_version_tracking_demo():
    """
    Run two versions of a prompt, track metrics separately.
    In production: route % of traffic to each version, compare quality.
    """
    print("\n=== Prompt Version A/B Tracking ===\n")

    store = ObservabilityStore()

    versions = {
        "v1_basic": {
            "system": "Classify alerts as P1/P2/P3. Return JSON: {\"severity\": \"P1|P2|P3\"}",
            "client": None
        },
        "v2_detailed": {
            "system": "You are a data engineering on-call classifier. Classify alerts into P1 (immediate), P2 (within 2h), P3 (next business day). Consider: data loss=P1, SLA breach=P2, warnings=P3. Return JSON: {\"severity\": \"P1|P2|P3\", \"reason\": \"one line\"}",
            "client": None
        }
    }

    for version in versions:
        versions[version]["client"] = ObservedAnthropicClient(store, prompt_version=version)

    alerts = [
        "Snowflake warehouse credit limit reached — all queries failing",
        "dbt model fct_orders ran 45 minutes, SLA is 30 minutes",
        "Fivetran sync completed with 0 new rows (expected ~50k)",
    ]

    for alert in alerts:
        for version, config in versions.items():
            config["client"].create(
                model="claude-haiku-4-5-20251001",
                max_tokens=64,
                temperature=0,
                system=config["system"],
                messages=[{"role": "user", "content": alert}]
            )

    print("By prompt version:")
    for version, metrics in store.by_prompt_version().items():
        print(f"\n  {version}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) and v else f"    {k}: {v}")


# ── Demo 3: Quality scoring integrated into pipeline ─────────────────────────

def quality_scoring_demo():
    """
    After each LLM call, run a quality check and attach the score.
    This gives you a time-series of output quality to alert on regression.
    """
    print("\n=== Quality Scoring Demo ===\n")

    store = ObservabilityStore()

    def call_and_score(prompt: str, expected_keys: List[str]) -> dict:
        start = time.time()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            temperature=0,
            system='Return only JSON with keys: severity, category, summary',
            messages=[{"role": "user", "content": prompt}]
        )
        latency = (time.time() - start) * 1000

        # Score the output
        text = response.content[0].text.strip()
        quality = 0.0
        try:
            parsed = json.loads(text)
            # Check all expected keys present and non-empty
            hits = sum(1 for k in expected_keys if parsed.get(k))
            quality = hits / len(expected_keys)
        except json.JSONDecodeError:
            quality = 0.0

        prices = COST_PER_TOKEN["claude-haiku-4-5-20251001"]
        obs = LLMObservation(
            call_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow().isoformat(),
            model="claude-haiku-4-5-20251001",
            prompt_version="v1.0",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency,
            ttft_ms=None,
            cost_usd=response.usage.input_tokens * prices["input"] + response.usage.output_tokens * prices["output"],
            success=True,
            quality_score=quality
        )
        store.record(obs)
        return {"output": text, "quality": quality}

    test_alerts = [
        "MySQL connection refused on replica host",
        "Revenue metric dropped 40% overnight",
        "Airflow scheduler heartbeat timeout",
    ]

    for alert in test_alerts:
        result = call_and_score(alert, expected_keys=["severity", "category", "summary"])
        icon = "✓" if result["quality"] >= 0.8 else "⚠"
        print(f"{icon} Alert: {alert[:50]}...")
        print(f"  Quality: {result['quality']:.0%} | Output: {result['output'][:80]}")

    avg_quality = sum(o.quality_score for o in store.observations if o.quality_score) / len(store.observations)
    print(f"\nAverage quality score: {avg_quality:.0%}")
    print(f"{'✓ Within acceptable range' if avg_quality > 0.8 else '⚠ Quality regression detected'}")


if __name__ == "__main__":
    basic_observability_demo()
    prompt_version_tracking_demo()
    quality_scoring_demo()
