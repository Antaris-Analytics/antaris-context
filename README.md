# antaris-context

**Zero-dependency context window optimization for AI agents.**

Manage context windows, token budgets, and message compression without external dependencies. Built for production AI agent systems that need deterministic, configurable context management.

## Install

```bash
pip install antaris-context
```

**Requirements:** Python 3.9+, no dependencies.

## Quick Start

```python
from antaris_context import ContextManager

# Initialize with 8K token budget
manager = ContextManager(total_budget=8000)

# Set section budgets
manager.set_section_budget('system', 1000)
manager.set_section_budget('memory', 2000) 
manager.set_section_budget('conversation', 4000)
manager.set_section_budget('tools', 1000)

# Add content with priorities
manager.add_content('system', "You are a helpful assistant.", priority='critical')
manager.add_content('memory', "User prefers concise responses.", priority='important')

# Add conversation messages with automatic compression and selection
messages = [
    {'role': 'user', 'content': 'What is Python?'},
    {'role': 'assistant', 'content': 'Python is a programming language...'},
    # ... more messages
]
manager.add_content('conversation', messages, priority='normal')

# Check usage
report = manager.get_usage_report()
print(f"Used: {report['total_used']}/{report['total_budget']} tokens")
print(f"Utilization: {report['utilization']:.1%}")

# Optimize context window
optimization = manager.optimize_context(target_utilization=0.85)
print(f"Optimization successful: {optimization['success']}")
```

## Core Components

### ContextManager

The main orchestrator. Manages budgets, applies strategies, and coordinates all optimization.

```python
from antaris_context import ContextManager

# Initialize with configuration file
manager = ContextManager(
    total_budget=8000,
    config_file='context_config.json'
)

# Add content with automatic strategy selection
manager.add_content('conversation', messages, priority='normal')

# Analyze and optimize
analysis = manager.analyze_context()
print(f"Efficiency score: {analysis['efficiency_score']:.2f}")

# Get optimization suggestions
for suggestion in analysis['optimization_suggestions']:
    print(f"- {suggestion['description']}")
```

### Content Selection Strategies

Choose what content to include when space is limited:

```python
# Recency strategy - newest first
manager.set_strategy('recency', prefer_high_priority=True)

# Relevance strategy - keyword matching
manager.set_strategy('relevance', min_score=0.2, case_sensitive=False)

# Hybrid strategy - combine recency and relevance
manager.set_strategy('hybrid', recency_weight=0.4, relevance_weight=0.6)

# Budget strategy - maximize value per token
manager.set_strategy('budget', approach='balanced')

# Use with query context
selected = manager.add_content('conversation', messages, query="Tell me about Python")
```

### Message Compression

Reduce token usage while preserving meaning:

```python
from antaris_context import MessageCompressor

# Configure compression levels
compressor = MessageCompressor('moderate')  # light, moderate, aggressive

# Compress individual messages
compressed = compressor.compress("This    has   lots  of\n\n\nwhitespace")
# Result: "This has lots of whitespace"

# Compress message lists
messages = [
    {'role': 'user', 'content': 'Very long message...'},
    {'role': 'tool', 'content': '500 lines of output...'}
]
compressed_msgs = compressor.compress_message_list(messages, max_content_length=500)

# Tool output compression (keep first/last N lines)
output = compressor.compress_tool_output(long_output, max_lines=20, keep_first=10, keep_last=10)

# Get compression stats
stats = compressor.get_compression_stats()
print(f"Saved {stats['bytes_saved']} bytes ({stats['compression_ratio']:.1%})")
```

### Context Analysis

Understand usage patterns and get optimization advice:

```python
from antaris_context import ContextProfiler

profiler = ContextProfiler(log_file='context_analysis.jsonl')
analysis = profiler.analyze_window(manager.window)

# Section analysis
for section, data in analysis['section_analysis'].items():
    print(f"{section}: {data['utilization']:.1%} utilized, {data['status']}")

# Waste detection
waste = analysis['waste_detection']
print(f"Found {len(waste['waste_items'])} waste sources")
print(f"Total waste: {waste['total_waste_tokens']} tokens")

# Budget reallocation suggestions
suggestions = profiler.suggest_budget_reallocation(manager.window)
for section, budget in suggestions['suggested_budgets'].items():
    print(f"{section}: {budget} tokens (was {suggestions['current_budgets'][section]})")

# Historical trends
trends = profiler.get_historical_trends(days=7)
print(f"Efficiency trend: {trends['efficiency_trend']['current']:.2f}")
```

## Configuration

Use JSON files for persistent configuration:

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
  "auto_compress": true,
  "profiler_log_file": "profiler.jsonl"
}
```

```python
# Load configuration
manager = ContextManager(config_file='config.json')

# Modify and save
manager.set_compression_level('aggressive')
manager.save_config('updated_config.json')
```

## Priority System

Content is prioritized for inclusion:

- **`critical`**: Never truncated, always included (system prompts, safety constraints)
- **`important`**: High priority, removed only when necessary (recent context, user preferences)  
- **`normal`**: Standard priority, balanced selection (conversation history)
- **`optional`**: First to be removed when space is needed (old messages, verbose outputs)

```python
# Add content with priorities
manager.add_content('system', 'Safety: Never generate harmful content', priority='critical')
manager.add_content('memory', 'User likes Python examples', priority='important')
manager.add_content('conversation', 'How do I use decorators?', priority='normal')
manager.add_content('tools', 'Debug output: verbose trace...', priority='optional')

# During truncation, optional content is removed first, critical content never
```

## Truncation Strategies

When content exceeds budget, different strategies decide what to remove:

```python
# Oldest first (default)
manager.config['truncation_strategy'] = 'oldest_first'

# Lowest priority first
manager.config['truncation_strategy'] = 'lowest_priority'

# Smart summary preservation
manager.config['truncation_strategy'] = 'smart_summary_markers'
```

## Token Estimation

Uses character-based approximation (~4 characters per token):

```python
from antaris_context import ContextWindow

window = ContextWindow()
tokens = window._estimate_tokens("Hello world")  # ~3 tokens

# This is an approximation for efficiency
# Real token counts vary by model and tokenizer
```

## Real-World Example

Complete agent context management:

```python
import json
from antaris_context import ContextManager

# Initialize agent context
manager = ContextManager(total_budget=8000)
manager.set_section_budget('system', 800)
manager.set_section_budget('memory', 1200) 
manager.set_section_budget('conversation', 5000)
manager.set_section_budget('tools', 1000)

# Set hybrid strategy for balanced selection
manager.set_strategy('hybrid', recency_weight=0.3, relevance_weight=0.7)

# Add system prompt
system_prompt = """You are a coding assistant. 
Rules:
- Always provide working examples
- Explain complex concepts simply
- Ask clarifying questions when needed"""

manager.add_content('system', system_prompt, priority='critical')

# Add user memory/preferences
memories = [
    "User is learning Python",
    "Prefers concise explanations", 
    "Working on web development project"
]
for memory in memories:
    manager.add_content('memory', memory, priority='important')

# Add conversation history (will be selected by strategy)
conversation = [
    {'role': 'user', 'content': 'How do I create a web API in Python?'},
    {'role': 'assistant', 'content': 'You can use Flask or FastAPI. Here\'s a Flask example:\n\n```python\nfrom flask import Flask\napp = Flask(__name__)\n\n@app.route("/api/hello")\ndef hello():\n    return {"message": "Hello World"}\n\nif __name__ == "__main__":\n    app.run(debug=True)\n```'},
    {'role': 'user', 'content': 'What about authentication?'},
    # ... more messages
]

# Add with query context for relevance scoring
current_query = "How do I add JWT authentication to my Flask API?"
manager.add_content('conversation', conversation, query=current_query)

# Add tool outputs
tool_output = """
Flask-JWT-Extended installed successfully
Dependencies: PyJWT, Flask, Werkzeug
Configuration options:
- JWT_SECRET_KEY: Required
- JWT_ACCESS_TOKEN_EXPIRES: Optional, defaults to 15 minutes  
- JWT_REFRESH_TOKEN_EXPIRES: Optional, defaults to 30 days
"""

manager.add_content('tools', tool_output, priority='normal')

# Optimize for target utilization
optimization = manager.optimize_context(
    query=current_query, 
    target_utilization=0.85
)

print(f"Optimization successful: {optimization['success']}")
print(f"Actions taken: {optimization['actions_taken']}")

# Get final usage report
report = manager.get_usage_report()
print(f"\nFinal utilization: {report['utilization']:.1%}")
print(f"Sections:")
for section, data in report['sections'].items():
    print(f"  {section}: {data['used']}/{data['budget']} tokens ({data['utilization']:.1%})")

# Analyze for insights
analysis = manager.analyze_context()
print(f"\nEfficiency score: {analysis['efficiency_score']:.2f}")
print("Optimization suggestions:")
for suggestion in analysis['optimization_suggestions']:
    print(f"  - {suggestion['description']}")

# Export state for persistence
state = manager.export_state()
with open('agent_context_state.json', 'w') as f:
    f.write(state)
```

## What It Doesn't Do

**antaris-context** is focused and honest about its limitations:

- **No actual tokenization**: Uses character-based approximation (~4 chars/token). For exact counts, integrate with your model's tokenizer.

- **No LLM calls**: Purely deterministic processing. Relevance scoring uses simple keyword matching, not semantic similarity.

- **No content generation**: Won't summarize or rewrite content. It selects, compresses, and truncates existing content only.

- **No model-specific optimization**: Token estimates work generally but aren't tuned for specific models (GPT-4, Claude, etc).

- **No automatic learning**: Doesn't adapt strategies based on usage patterns. Configuration is explicit and static.

- **No distributed contexts**: Manages single context windows. For multi-agent or distributed scenarios, use multiple managers.

- **Limited compression**: Focuses on whitespace and structural compression, not semantic compression or paraphrasing.

This is intentional. The library does one thing well: deterministic context window management with configurable strategies.

## Design Philosophy

Built on principles proven by the antaris-* suite:

- **Zero dependencies**: Only Python stdlib. No version conflicts, minimal security surface.
- **File-based config**: JSON configuration for reproducible behavior across environments.
- **Deterministic**: Same inputs always produce same outputs. No randomness, no API calls.
- **Honest limitations**: Clear about what it does and doesn't do. No overselling.
- **Production-ready**: Designed for real agent systems, not demos or experiments.

## Performance

Rough benchmarks on modern hardware:

- **Token estimation**: ~100K characters/second
- **Message compression**: ~50K characters/second  
- **Strategy selection**: ~10K messages/second
- **Context analysis**: ~1K content items/second

Memory usage scales linearly with content size. No significant overhead for large contexts.

## Comparison

Similar libraries and how **antaris-context** differs:

| Library | Dependencies | Config | Deterministic | Token Counting | Strategies |
|---------|-------------|---------|---------------|---------------|------------|
| **antaris-context** | ✅ None | ✅ JSON files | ✅ Yes | ⚠️ Approximation | ✅ Pluggable |
| tiktoken | ✅ Minimal | ❌ Code only | ✅ Yes | ✅ Exact | ❌ None |
| langchain | ❌ Heavy | ❌ Code only | ❌ No | ✅ Via tiktoken | ⚠️ Limited |
| guidance | ❌ Heavy | ❌ Code only | ⚠️ Partial | ✅ Via transformers | ❌ None |

Choose **antaris-context** when you need:
- Zero-dependency deployment
- File-based configuration  
- Deterministic behavior
- Production-ready context management
- Pluggable selection strategies

Choose alternatives when you need:
- Exact tokenization for specific models
- Semantic similarity (use embeddings)
- Content summarization (use LLMs)
- Complex multi-modal contexts

## Contributing

This library is part of the antaris-* suite. Contributions welcome:

1. Keep zero dependencies
2. Maintain deterministic behavior  
3. Add tests for new features
4. Update documentation
5. Follow existing code style

## License

Apache 2.0 - see [LICENSE](LICENSE) file.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.