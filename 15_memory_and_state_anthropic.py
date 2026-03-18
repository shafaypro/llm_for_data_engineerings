"""
15_memory_and_state_anthropic.py
==================================
Concept: Memory & State Management
------------------------------------
LLMs have NO memory between API calls. Every call starts fresh.
To build stateful applications (chatbots, long-running agents, pipelines
that learn over time), you must manage state yourself.

Four memory strategies:
1. In-context memory   — stuff everything into the context window (simple, limited)
2. Sliding window      — keep last N turns, drop older ones
3. Summarisation       — compress old turns into a summary, keep recent ones
4. External memory     — store facts in a DB/file, retrieve when needed (persistent)

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json
from pathlib import Path
from datetime import datetime

client = anthropic.Anthropic()


# ── Strategy 1: In-context (naive) ────────────────────────────────────────────

def strategy_1_full_history():
    """
    Keep the entire conversation in the messages array.
    Simple but hits the context limit for long conversations.
    Every token in history costs money on every call.
    """
    print("=== Strategy 1: Full History in Context ===\n")

    history = []

    def chat(user_message: str) -> str:
        history.append({"role": "user", "content": user_message})
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system="You are a data engineering assistant. Be concise.",
            messages=history
        )
        answer = response.content[0].text.strip()
        history.append({"role": "assistant", "content": answer})
        print(f"Tokens used: {response.usage.input_tokens} input (grows every turn)")
        return answer

    turns = [
        "My Airflow DAG is called daily_orders_etl",
        "It runs at 2am UTC and usually takes 20 minutes",
        "What's a good SLA I should set for it?",
        "And what alert threshold makes sense for that SLA?",
    ]

    for turn in turns:
        print(f"User: {turn}")
        answer = chat(turn)
        print(f"Claude: {answer[:150]}\n")


# ── Strategy 2: Sliding window ────────────────────────────────────────────────

def strategy_2_sliding_window():
    """
    Keep only the last N turns. Oldest turns are dropped when limit is hit.
    Trade-off: model loses early context but token cost stays bounded.
    Good for: chat interfaces, interactive tools.
    """
    print("\n=== Strategy 2: Sliding Window (last 4 turns) ===\n")

    WINDOW_SIZE = 4  # keep last 4 messages (2 turns)
    history = []

    def chat(user_message: str) -> str:
        history.append({"role": "user", "content": user_message})

        # Trim to window — always keep pairs to maintain user/assistant alternation
        windowed = history[-WINDOW_SIZE:] if len(history) > WINDOW_SIZE else history

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system="You are a data engineering assistant. Be concise.",
            messages=windowed
        )
        answer = response.content[0].text.strip()
        history.append({"role": "assistant", "content": answer})

        dropped = max(0, len(history) - WINDOW_SIZE)
        print(f"History: {len(history)} total, {len(windowed)} in window, {dropped} dropped")
        return answer

    turns = [
        "I use Snowflake for my warehouse",
        "My main fact table is fct_orders",
        "What partition strategy should I use for fct_orders?",
        "How do I monitor partition pruning effectiveness?",
        "Can you remind me which warehouse I use?",  # will lose early context
    ]

    for turn in turns:
        print(f"User: {turn}")
        answer = chat(turn)
        print(f"Claude: {answer[:120]}\n")


# ── Strategy 3: Summarisation ─────────────────────────────────────────────────

def strategy_3_summarisation():
    """
    When history grows too long, compress old turns into a summary paragraph.
    Keep the summary + recent turns. Better recall than sliding window.
    Good for: long sessions where early context matters.
    """
    print("\n=== Strategy 3: Summarisation ===\n")

    SUMMARISE_THRESHOLD = 6  # summarise when we hit this many messages
    history = []
    summary = ""

    def summarise_history(messages: list) -> str:
        """Compress conversation history into a summary."""
        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": f"""Summarise this conversation in 3-4 sentences.
Focus on: decisions made, facts established, context that would help continue the conversation.

<conversation>
{history_text}
</conversation>"""}]
        )
        return response.content[0].text.strip()

    def chat(user_message: str) -> str:
        nonlocal summary, history

        # Build system prompt with summary if we have one
        system = "You are a data engineering assistant."
        if summary:
            system += f"\n\nContext from earlier in this conversation:\n{summary}"

        # Summarise if history is getting long
        if len(history) >= SUMMARISE_THRESHOLD:
            print(f"  [Summarising {len(history)} messages...]")
            summary = summarise_history(history)
            print(f"  [Summary: {summary[:100]}...]")
            history = history[-2:]  # keep only last 2 messages after summarisation

        history.append({"role": "user", "content": user_message})
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=system,
            messages=history
        )
        answer = response.content[0].text.strip()
        history.append({"role": "assistant", "content": answer})
        return answer

    turns = [
        "I'm building a pipeline for a fintech startup",
        "We process about 50k transactions per day",
        "Our main pain point is late-arriving data from banking APIs",
        "We're using Airflow on AWS ECS",
        "What's the best pattern for handling late data in Airflow?",
        "How would I implement that with Snowflake?",
        "Given everything I've told you about our setup, what should I build first?",  # needs early context
    ]

    for turn in turns:
        print(f"User: {turn}")
        answer = chat(turn)
        print(f"Claude: {answer[:150]}\n")


# ── Strategy 4: External memory (persistent across sessions) ──────────────────

def strategy_4_external_memory():
    """
    Store facts about the user/system in a persistent file or database.
    Retrieve relevant facts at the start of each session and inject into context.
    This is how you give an AI assistant 'memory' that persists across restarts.

    In production: replace file storage with Redis, DynamoDB, or Postgres.
    """
    print("\n=== Strategy 4: External Memory ===\n")

    MEMORY_FILE = Path("/tmp/data_engineer_memory.json")

    def load_memory() -> dict:
        if MEMORY_FILE.exists():
            return json.loads(MEMORY_FILE.read_text())
        return {"facts": [], "preferences": {}, "context": {}}

    def save_memory(memory: dict) -> None:
        MEMORY_FILE.write_text(json.dumps(memory, indent=2))

    def extract_and_store_facts(message: str, memory: dict) -> dict:
        """Use Claude to extract storable facts from user messages."""
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0,
            system="""Extract persistent facts from user messages that should be remembered for future sessions.
Return JSON: {"facts": ["fact1", "fact2"], "preferences": {"key": "value"}, "context": {"key": "value"}}
Only extract concrete, reusable facts. Return empty lists/dicts if nothing useful to store.
Return ONLY JSON.""",
            messages=[{"role": "user", "content": f"Extract facts from: {message}"}]
        )
        try:
            new_info = json.loads(response.content[0].text.strip())
            memory["facts"].extend(new_info.get("facts", []))
            memory["facts"] = list(set(memory["facts"]))[-20:]  # keep last 20 unique facts
            memory["preferences"].update(new_info.get("preferences", {}))
            memory["context"].update(new_info.get("context", {}))
        except Exception:
            pass
        return memory

    def chat_with_memory(user_message: str) -> str:
        memory = load_memory()

        # Build context from memory
        memory_context = ""
        if memory["facts"]:
            memory_context += "Known facts about this user's setup:\n" + "\n".join(f"- {f}" for f in memory["facts"])
        if memory["preferences"]:
            memory_context += f"\nPreferences: {memory['preferences']}"

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=f"""You are a persistent data engineering assistant.
{f"Memory from previous sessions:{chr(10)}{memory_context}" if memory_context else ""}
Use stored context to give personalised answers.""",
            messages=[{"role": "user", "content": user_message}]
        )
        answer = response.content[0].text.strip()

        # Extract and store new facts
        memory = extract_and_store_facts(user_message, memory)
        memory["last_seen"] = datetime.utcnow().isoformat()
        save_memory(memory)

        return answer

    # Simulate two separate sessions
    print("--- Session 1 ---")
    turns_s1 = [
        "I work at a startup, we use dbt + Snowflake + Airflow",
        "I prefer concise answers without fluff",
        "Our biggest table has 500M rows",
    ]
    for turn in turns_s1:
        print(f"User: {turn}")
        answer = chat_with_memory(turn)
        print(f"Claude: {answer[:120]}\n")

    print("--- Session 2 (new Python process, memory persisted) ---")
    turns_s2 = [
        "Based on what you know about my setup, what indexing strategy should I use?",
    ]
    for turn in turns_s2:
        print(f"User: {turn}")
        answer = chat_with_memory(turn)
        print(f"Claude: {answer[:250]}\n")

    # Show what was stored
    memory = load_memory()
    print(f"Stored facts: {memory['facts']}")
    print(f"Stored preferences: {memory['preferences']}")


if __name__ == "__main__":
    strategy_1_full_history()
    strategy_2_sliding_window()
    strategy_3_summarisation()
    strategy_4_external_memory()
