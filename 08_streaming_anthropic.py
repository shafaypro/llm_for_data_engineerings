"""
08_streaming_anthropic.py
=========================
Concept: Streaming
------------------
Streaming returns tokens as they're generated instead of waiting for the
full response. For a 500-token response at 40 tokens/sec, streaming means
the user sees output in ~0.1s instead of waiting 12.5s for the full batch.

Think of it like Kafka vs batch ETL — you process events as they arrive
rather than waiting for the full file.

When to use streaming:
- Any user-facing tool (Slack bots, CLIs, web apps)
- Long responses where early content is useful (reports, explanations)

When NOT to stream:
- Background pipeline jobs (latency doesn't matter)
- When you need to parse the full response before using it (JSON output)
- Batch processing thousands of requests

Install: pip install anthropic
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import time
import sys

client = anthropic.Anthropic()


# ── Demo 1: Basic streaming ────────────────────────────────────────────────────
def basic_streaming_demo():
    """Stream tokens to stdout as they arrive."""

    print("=== Basic Streaming Demo ===")
    print("Response (streaming): ", end="", flush=True)

    start = time.time()
    first_token_time = None

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": "Explain what a slowly changing dimension is in data warehousing. Be concise."
        }]
    ) as stream:
        for text in stream.text_stream:
            if first_token_time is None:
                first_token_time = time.time()
                print()  # newline after "Response:"
            print(text, end="", flush=True)

    total_time = time.time() - start
    print(f"\n\nTime to first token: {first_token_time - start:.2f}s")
    print(f"Total time: {total_time:.2f}s")

    # Get final message with usage stats
    final_message = stream.get_final_message()
    print(f"Output tokens: {final_message.usage.output_tokens}")
    print(f"Effective tokens/sec: {final_message.usage.output_tokens / total_time:.1f}")


# ── Demo 2: Streaming vs non-streaming latency comparison ─────────────────────
def latency_comparison_demo():
    """
    Show the difference in perceived latency between streaming and non-streaming.
    The total tokens generated is the same — streaming just shows them earlier.
    """

    prompt = "List 5 best practices for optimising dbt model performance. Be detailed."

    print("\n=== Latency Comparison Demo ===\n")

    # Non-streaming: wait for everything
    print("Non-streaming (waiting for full response)...")
    start = time.time()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    non_stream_time = time.time() - start
    print(f"  Wait time before seeing anything: {non_stream_time:.2f}s")
    print(f"  First 100 chars: {response.content[0].text[:100]}...")

    print()

    # Streaming: tokens appear immediately
    print("Streaming (tokens appear as generated)...")
    start = time.time()
    first_token_t = None
    char_count = 0

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            if first_token_t is None:
                first_token_t = time.time()
            char_count += len(text)

    stream_total = time.time() - start
    print(f"  Time to first token: {first_token_t - start:.2f}s")
    print(f"  Total time (same as non-stream): {stream_total:.2f}s")
    print(f"\n  Perceived improvement: {non_stream_time - (first_token_t - start):.2f}s faster first response")


# ── Demo 3: Streaming with event hooks ────────────────────────────────────────
def streaming_with_events_demo():
    """
    Use the event-based streaming API for more control —
    useful when you need to handle different event types separately
    (e.g. log usage stats, detect tool calls, handle errors).
    """

    print("\n=== Streaming with Event Hooks ===\n")

    token_count = 0
    full_text = []

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": "What is partitioning in Snowflake and when should I use it?"}]
    ) as stream:
        for event in stream:
            event_type = type(event).__name__

            if event_type == "RawContentBlockDeltaEvent":
                if hasattr(event.delta, "text"):
                    full_text.append(event.delta.text)
                    token_count += 1
                    # In a real app: push to WebSocket, write to Slack partial message, etc.

            elif event_type == "RawMessageStopEvent":
                print("Stream complete.")

        final = stream.get_final_message()

    print("".join(full_text))
    print(f"\nFinal usage: {final.usage.input_tokens} in / {final.usage.output_tokens} out")


# ── Demo 4: Streaming Slack bot pattern ───────────────────────────────────────
def slack_bot_pattern_demo():
    """
    In a real Slack bot, you'd:
    1. Post an initial message with a spinner
    2. Update it with each chunk as they stream in
    3. Edit to final message when complete

    This demo simulates that with console output.
    """

    print("\n=== Slack Bot Streaming Pattern ===\n")

    question = "The fct_orders model is running slowly. What should I check first?"

    # Simulate Slack message updates
    print("Posting to #data-engineering...")
    print("Bot: [thinking...]", end="", flush=True)

    accumulated = ""
    update_interval = 50  # update every N chars (in real Slack: every 1-2 seconds)

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="You are a helpful data engineering assistant. Be concise and actionable.",
        messages=[{"role": "user", "content": question}]
    ) as stream:
        for text in stream.text_stream:
            accumulated += text
            # Simulate updating the Slack message
            if len(accumulated) % update_interval < len(text):
                print(f"\rBot: {accumulated[:80]}{'...' if len(accumulated) > 80 else ''}", end="", flush=True)

    print(f"\rBot: {accumulated}")
    print("\n[Message finalised in Slack]")


if __name__ == "__main__":
    basic_streaming_demo()
    latency_comparison_demo()
    streaming_with_events_demo()
    slack_bot_pattern_demo()
