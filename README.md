# antaris-context

**Zero-dependency context window optimization for AI agents.**

**v3.1.0** — Budget-aware compression • Compaction integration • Lossless prioritization • 8 source files • 150 tests**

## What's New in v3.1.0

- **Budget-Aware Compression** — Keep high-priority memories intact; drop lowest-value items when budget exceeded. Never silently truncate mid-entry.
- **Compaction Integration** — Context state injected cleanly post-compaction. Full conversation continuity even after session reset.
- **Lossless Prioritization** — Every context item carries importance + urgency. System respects your priorities.
- **Hybrid Compression** — Try compression on each section. Fall back to uncompressed if compression fails.
- **Zero Dependencies** — Pure Python stdlib. Safe to import anywhere without version conflicts.

## Phase 4 Roadmap

**v3.2:** Adaptive compression based on available tokens (auto-tune compression level)  
**v4.0:** Cold vs warm session awareness (inject more context on first turn, less mid-conversation)  
**v4.1+:** Semantic importance scoring (memories that matter more semantically get protected first)

Manage context windows, token budgets, turn lifecycle, and message compression without external dependencies. Integrates with `antaris-memory` for memory-informed priority boosting and `antaris-router` for adaptive budget allocation. Built for production AI agent systems that need deterministic, configurable context management.

[![PyPI](https://img.shields.io/pypi/v/antaris-context)](https://pypi.org/project/antaris-context/)
[![Tests](https://github.com/Antaris-Analytics/antaris-context/actions/workflows/tests.yml/badge.svg)](https://github.com/Antaris-Analytics/antaris-context/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

---

## Install

```bash
pip install antaris-context
```

---

## Quick Start

```python
from antaris_context import ContextManager

# Initialize with a token budget and template
manager = ContextManager(total_budget=8000, template="code_assistant")

# Add conversation turns
manager.add_turn("user", "How do I add JWT auth to my Flask API?")
manager.add_turn("assistant", "Use flask-jwt-extended. Here's a minimal example...")

# Check usage
print(f"Turns: {manager.turn_count}")
report = manager.get_usage_report()
print(f"Used: {report['total_used']}/{report['total_budget']} tokens ({report['utilization']:.1%})")

# Render for your LLM provider
messages = manager.render(format="anthropic")   # Anthropic message format
messages = manager.render(format="openai")      # OpenAI message format
plain    = manager.render(format="raw")         # "role: content\n..." string
```

---

## Token Budgets

Every `ContextManager` splits its total budget across four sections — `system`, `memory`, `conversation`, and `tools`. Built-in templates pre-allocate these for common patterns:

```python
templates = ContextManager.get_available_templates()
# {
#   'chatbot':          {'system': 800,  'memory': 1500, 'conversation': 5000, 'tools': 700},
#   'agent_with_tools': {'system': 1200, 'memory': 2000, 'conversation': 3500, 'tools': 1300},
#   'rag_pipeline':     {'system': 600,  'memory': 1000, 'conversation': 4500, 'tools': 1900},
#   'code_assistant':   {'system': 1000, 'memory': 1800, 'conversation': 4000, 'tools': 1200},
#   'balanced':         {'system': 1000, 'memory': 2000, 'conversation': 4000, 'tools': 1000},
# }

manager = ContextManager(total_budget=8000, template="agent_with_tools")
manager.apply_template("rag_pipeline")  # Switch template mid-session
```

Add content to sections with priority levels:

```python
manager.add_content('system', "You are a coding assistant.", priority='critical')
manager.add_content('memory', "User prefers Python examples.", priority='important')
manager.add_content('conversation', messages, priority='normal')
manager.add_content('tools', long_debug_output, priority='optional')

# Priority levels:
# critical  — never truncated (system prompts, safety rules)
# important — removed only when necessary
# normal    — standard selection (conversation history)
# optional  — first to go when space is needed
```

---

## Turn Lifecycle

Turns are the core unit of conversation management. Add turns, compact old ones, and render for any provider:

```python
manager = ContextManager(total_budget=16000, template="agent_with_tools")

# Add turns from a conversation
for msg in conversation_history:
    manager.add_turn(msg["role"], msg["content"])

print(f"Turn count: {manager.turn_count}")

# Configure retention policy
manager.set_retention_policy(
    keep_last_n_verbatim=10,
    summarize_older=True,
    max_turns=100,
)

# Compact old turns (truncates to 120 chars by default)
removed = manager.compact_older_turns(keep_last=20)
print(f"Compacted {removed} turns")

# Plug in a real summarizer for semantic compression
def my_summarizer(text: str) -> str:
    # Call your LLM to summarize
    return "Summary: ..."

manager.set_summarizer(my_summarizer)
manager.compact_older_turns(keep_last=20)
# Older turns are now passed to my_summarizer instead of truncated
```

---

## Rendering

`render()` produces provider-ready message lists. Use the `format` parameter (or the `provider` alias):

```python
# Anthropic format — list of {"role": ..., "content": ...} dicts
messages = manager.render(format="anthropic")

# OpenAI format — same structure, ready for chat completions
messages = manager.render(format="openai")

# Raw format — plain text string "role: content\n..."
plain = manager.render(format="raw")

# Inject a system prompt
messages = manager.render(format="openai", system_prompt="Be concise.")
```

---

## Sliding Window

`get_sliding_window()` returns the most recent turns without mutating state — a lightweight read-only view:

```python
# Get the last 5 turns
recent = manager.get_sliding_window(max_turns=5)
# [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

# Use it for a focused context window before an API call
for turn in recent:
    print(f"{turn['role']}: {turn['content'][:80]}")
```

---

## Trim to Budget

`trim_to_budget()` prunes section content (lowest priority first) until the total token count is within the target:

```python
manager.add_content('system', system_prompt, priority='critical')
manager.add_content('conversation', long_history, priority='normal')
manager.add_content('tools', tool_output, priority='optional')

# Hard-cap total tokens at 4000
freed = manager.trim_to_budget(max_tokens=4000)
print(f"Freed {freed} tokens")
# 'optional' items are dropped first, then 'normal', then 'important'
# 'critical' items are never removed
```

---

## Overflow Cascade

When a section exceeds its budget, `cascade_overflow()` borrows unused slack from other sections:

```python
manager.add_content('tools', very_long_tool_output, priority='normal')

if manager.is_over_budget():
    redistributed = manager.cascade_overflow('tools')
    print(f"Borrowed {redistributed} tokens from other sections")
```

---

## Adaptive Compression

The `optimize_context()` method applies multi-stage compression — content-level compression, priority-aware truncation, and section rebalancing — to hit a target utilization:

```python
result = manager.optimize_context(query="JWT authentication", target_utilization=0.85)

print(f"Success: {result.success}")
print(f"Tokens saved: {result.tokens_saved}")
print(f"Compression ratio: {result.compression_ratio:.2f}")
print(f"Actions: {result.actions_taken}")
```

Set compression levels and selection strategies:

```python
manager.set_compression_level('aggressive')  # light, moderate, aggressive

manager.set_strategy('hybrid', recency_weight=0.4, relevance_weight=0.6)
manager.set_strategy('recency', prefer_high_priority=True)
manager.set_strategy('budget', approach='balanced')
```

Track usage patterns over time and let budgets adapt automatically:

```python
manager.enable_adaptive_budgets(target_utilization=0.85)
manager.track_usage()

suggestions = manager.suggest_adaptive_reallocation()
if suggestions:
    for section, budget in suggestions['suggested_budgets'].items():
        current = suggestions['current_budgets'][section]
        print(f"{section}: {current} -> {budget} tokens")

manager.apply_adaptive_reallocation(auto_apply=True)
```

---

## Relevance Scoring

Content can be scored for relevance against a query, so the most relevant items survive compression:

```python
# Add content with a query for relevance-based selection
manager.add_content('conversation', messages, query="JWT authentication Flask")

# The hybrid strategy weighs recency and relevance together
manager.set_strategy('hybrid', recency_weight=0.4, relevance_weight=0.6)
```

The `ImportanceWeightedCompressor` scores items on five axes — recency, priority label, explicit importance score, message role, and content density — then keeps top-N verbatim, compresses middle items, and drops low-scorers:

```python
from antaris_context import ImportanceWeightedCompressor

iwc = ImportanceWeightedCompressor(keep_top_n=5, compress_middle=True, drop_threshold=0.1)
result = iwc.compress_items(content_items)
# result["kept"]       — items kept verbatim
# result["compressed"] — items with condensed content
# result["dropped"]    — items removed entirely
# result["tokens_saved"]
```

The `SemanticChunker` splits text at sentence and paragraph boundaries with configurable chunk sizes:

```python
from antaris_context import SemanticChunker

chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=500)
chunks = chunker.chunk(long_text)
for chunk in chunks:
    print(f"[{chunk.chunk_type}] importance={chunk.importance_score:.2f}  {chunk.content[:60]}...")
```

---

## Suite Integration

### Memory-informed priority boosting

Connect `antaris-memory` so that `optimize_context()` automatically boosts content items matching recent memory hits:

```python
from antaris_context import ContextManager
from antaris_memory import MemorySystem

mem = MemorySystem("./workspace")
mem.load()

manager = ContextManager(total_budget=8000)
manager.set_memory_client(mem)
# optimize_context() now elevates items that overlap with memory search results
```

### Router-driven budget adaptation

Accept hints from `antaris-router` to shift section budgets and target utilization on the fly:

```python
from antaris_router import Router

router = Router(config_path="./config")
result = router.route(user_input)

manager.set_router_hints(result.routing_hints)
# Hints like {'boost_section': 'tools', 'target_utilization': 0.7}
# shift 10% of total budget to the boosted section automatically
```

---

## Compression Utilities

```python
from antaris_context import MessageCompressor

compressor = MessageCompressor('moderate')  # light, moderate, aggressive

# Compress a single string
compressed = compressor.compress(long_text)

# Compress a list of message dicts
compressed_msgs = compressor.compress_message_list(messages, max_content_length=500)

# Compress tool output (keep first/last N lines)
output = compressor.compress_tool_output(long_output, max_lines=20, keep_first=10, keep_last=10)

# Check stats
stats = compressor.get_compression_stats()
print(f"Saved {stats['bytes_saved']} bytes ({stats['compression_ratio']:.1%})")
```

Large inputs (>2 MB) trigger a warning — chunk before compressing for best performance.

---

## Cross-Session Snapshots

Save and restore full context state (including content) across sessions:

```python
# Export to a JSON-serializable dict
snapshot = manager.export_snapshot(include_importance_above=0.5)

# Reconstruct a new manager from the snapshot
manager2 = ContextManager.from_snapshot(snapshot)

# In-process structural snapshots (lightweight, no content)
manager.save_snapshot("pre-refactor")
manager.restore_snapshot("pre-refactor")

for info in manager.list_snapshots():
    print(info['name'])
```

---

## Context Analysis

```python
analysis = manager.analyze_context()
print(f"Efficiency score: {analysis['efficiency_score']:.2f}")

for section, data in analysis['section_analysis'].items():
    print(f"{section}: {data['utilization']:.1%} — {data['status']}")

for suggestion in analysis['optimization_suggestions']:
    print(f"  - {suggestion['description']}")
```

---

## Complete Agent Example

```python
from antaris_context import ContextManager

manager = ContextManager(total_budget=8000, template="code_assistant")

# System prompt (never truncated)
manager.add_content('system',
    "You are a coding assistant. Always provide working examples.",
    priority='critical')

# User memory
for memory in ["User is learning Python", "Prefers concise explanations"]:
    manager.add_content('memory', memory, priority='important')

# Conversation turns
for turn in conversation_history:
    manager.add_turn(turn["role"], turn["content"])

# Compact if needed
if manager.is_over_budget():
    manager.compact_older_turns(keep_last=20)

# Hard-trim to budget
manager.trim_to_budget(max_tokens=7500)

# Optimize to target utilization
result = manager.optimize_context(query=current_query, target_utilization=0.85)

# Render for your provider
messages = manager.render(format="openai")
response = openai_client.chat.completions.create(model="gpt-4o", messages=messages)
```

---

## Configuration File

```json
{
  "compression_level": "moderate",
  "strategy": "hybrid",
  "strategy_params": {
    "recency_weight": 0.4,
    "relevance_weight": 0.6
  },
  "section_budgets": {
    "system": 1000,
    "memory": 2000,
    "conversation": 4000,
    "tools": 1000
  },
  "truncation_strategy": "oldest_first",
  "auto_compress": true
}
```

```python
manager = ContextManager(config_file="config.json")
manager.set_compression_level("aggressive")
manager.save_config("updated_config.json")
```

---

## Token Estimation

Uses character-based approximation (~4 chars/token). Fast and sufficient for budget management. For exact counts, plug in your model's tokenizer:

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
manager._estimate_tokens = lambda text: len(enc.encode(text))
```

---

## What It Doesn't Do

- **No actual tokenization** — character-based approximation. Plug in your tokenizer for exact counts.
- **No LLM calls** — purely deterministic. The pluggable `set_summarizer()` is optional; without it, compaction is structural only.
- **No content generation** — selects, compresses, and truncates existing content. Won't paraphrase.
- **No distributed contexts** — manages single context windows. For multi-agent scenarios, use multiple managers.

---

## Running Tests

```bash
git clone https://github.com/Antaris-Analytics/antaris-context.git
cd antaris-context
python -m pytest tests/ -v
```

All 150 tests pass with zero external dependencies.

---

## Part of the Antaris Analytics Suite — v3.0.0

- **[antaris-memory](https://pypi.org/project/antaris-memory/)** — Persistent memory for AI agents
- **[antaris-router](https://pypi.org/project/antaris-router/)** — Adaptive model routing with SLA enforcement
- **[antaris-guard](https://pypi.org/project/antaris-guard/)** — Security and prompt injection detection
- **antaris-context** — Context window optimization (this package)
- **[antaris-pipeline](https://pypi.org/project/antaris-pipeline/)** — Agent orchestration pipeline
- **[antaris-contracts](https://pypi.org/project/antaris-contracts/)** — Versioned schemas, failure semantics, and debug CLI

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
