# Roadmap

## v0.1.0 (current)
- Token budget allocation across context sections
- Per-section tracking with overflow detection
- Message compression at 3 levels (light/moderate/aggressive)
- 4 strategies: Recency, Relevance, Hybrid, Budget
- Context profiler with waste detection and optimization suggestions
- 56 tests, zero dependencies

## v0.2.0
- Provider-aware token counting — handle OpenAI vs Anthropic tokenizer differences
- Section templates — prebuilt configurations for common agent patterns
- Import/export — serialize context state for session handoff
- Improved compression — smarter sentence selection, better summary generation

## v0.5.0
- Adaptive budgets — learn from usage patterns which sections matter most
- Priority cascading — automatically redistribute budget when sections overflow
- Context snapshots — save/restore context state at checkpoints
- Metrics and telemetry — track compression ratios, waste percentages over time

## v1.0.0
- Production hardening based on real-world usage feedback
- Performance benchmarks and optimization
- Integration testing with major LLM providers
- Full review cycle (Claude + GPT-5.2) at production quality bar
