# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-02-16

### Fixed (Claude Review Round 1)
- **🔴 Sentence selection O(n²) dedup bug** — `_smart_sentence_selection` used `list.index()` which returns first occurrence for duplicates, corrupting order. Now tracks original index from the start.
- **🔴 `_truncate_lowest_priority` mutation during iteration** — inner loop mutated lists while outer priority loop continued, risking over-removal. Rewritten to collect candidates first, then rebuild in a single pass.
- **🔴 `apply_adaptive_reallocation` ignored `auto_apply=False`** — would auto-apply when `potential_savings > 100` regardless of flag. Now strictly respects the parameter.
- **🟡 `_estimate_tokens` stripped whitespace** before counting — whitespace consumes tokens. Removed `.strip()`.
- **🟡 `from_json` didn't restore `used` counts** — deserialization silently dropped usage data. Now restores `used` for roundtrip fidelity.
- **🟡 `get_section_utilization` returned `float('inf')`** for zero-budget sections with content — now returns `1.0`.
- **🟡 `HybridStrategy` used private `_extract_keywords`/`_calculate_relevance`** — promoted to public API on `RelevanceStrategy` to decouple implementations.
- **🟢 Bare `except:` in profiler** `_parse_timestamp` — narrowed to `(ValueError, AttributeError, TypeError)`.
- **🟢 `re` import** in profiler moved from function-level to module-level.

### Improved (Claude Review Round 1)
- **`cascade_overflow` docstring** clarified: transfers unused budget slack only, never displaces content.
- 21 new tests covering cascade overflow, snapshots, adaptive budgets, templates, sentence dedup regression, `from_json` roundtrip, utilization bounds.
- **77 tests total** (up from 56).

## [1.0.0] - 2026-02-16

### 🎉 Major Release - Production Ready

#### 🐛 Bug Fixes
- **Fixed atomic file writes**: `save_config()` now uses atomic_write_json pattern (mkstemp + fsync + os.replace) instead of raw open()
- **Fixed error logging**: `_log_analysis()` in profiler now logs warnings instead of silently swallowing errors
- **Fixed directory creation**: Added guard against empty dirname in file path operations
- **Fixed index corruption**: `_truncate_oldest_first` algorithm completely rewritten to prevent index corruption during list manipulation
- **Documented token estimation**: Clearly documented the len(text)//4 approximation with notes about provider-aware counting limitations

#### ✨ New Features

##### Section Templates (v0.2.0)
- **Prebuilt configurations** for common agent patterns:
  - `chatbot`: Basic chatbot with balanced memory and conversation space
  - `agent_with_tools`: AI agent with emphasis on tool definitions and working memory
  - `rag_pipeline`: RAG system optimized for retrieval and Q&A
  - `code_assistant`: Code-focused with space for analysis and formatting tools
  - `balanced`: Equal distribution across all sections
- **Template application**: `apply_template(template_name)` method and constructor `template` parameter
- **Template discovery**: `get_available_templates()` class method

##### Improved Compression (v0.2.0)
- **Smart sentence selection** in aggressive mode: intelligently ranks and selects most important sentences
- **Sentence scoring algorithm**: considers position, length, content indicators, questions, numbers, and filters filler words
- **Target ratio control**: configurable compression ratio (default 70% of sentences retained)

##### Adaptive Budgets (v0.5.0)
- **Usage tracking**: `track_usage()` method captures utilization patterns over time
- **Smart reallocation**: `suggest_adaptive_reallocation()` analyzes patterns and suggests optimal budget distribution
- **Auto-application**: `apply_adaptive_reallocation()` with configurable thresholds
- **Historical analysis**: maintains rolling window of last 100 usage snapshots

##### Priority Cascading (v0.5.0)  
- **Overflow redistribution**: `cascade_overflow()` automatically redistributes budget from overflowing sections
- **Available budget detection**: finds sections with spare capacity and reallocates intelligently
- **Budget balancing**: maintains overall budget constraints while optimizing utilization

##### Context Snapshots (v0.5.0)
- **State checkpointing**: `save_snapshot(name)` captures complete context state
- **Restoration**: `restore_snapshot(name)` restores previous state including configuration and window contents
- **Snapshot management**: `list_snapshots()` for discovering saved states
- **Use cases**: A/B testing, rollback scenarios, experimental configurations

##### Performance Benchmarks (v1.0.0)
- **Comprehensive benchmark suite** (`benchmarks/bench_context.py`) measures ops/sec for all major operations
- **14 benchmark categories**: window operations, compression levels, strategies, manager operations, profiler, snapshots, adaptive features
- **Statistical analysis**: warmup iterations, std deviation, min/max times
- **JSON output**: results saved to `benchmark_results.json` for tracking regressions
- **Performance baseline**: 2.8M+ combined ops/sec on modern hardware

##### Atomic Writes Everywhere (v1.0.0)
- **New utils.py module**: centralized atomic file operations following antaris-guard patterns
- **Safe configuration saves**: all JSON writes now use atomic operations with fsync
- **Consistency guarantees**: prevents partial writes and corruption during concurrent access
- **Error handling**: proper logging instead of silent failures

#### 🔧 Infrastructure Improvements
- **Python 3.9+ compatibility**: tested on all supported versions
- **Zero dependencies**: maintains stdlib-only approach
- **Enhanced documentation**: token estimation limitations clearly documented
- **Improved error messages**: better context for debugging issues

### 🧪 Testing
- **All 56 tests pass**: comprehensive test coverage maintained
- **New benchmark infrastructure**: performance regression detection
- **Real-world test data**: benchmarks use realistic content patterns

### 🚀 Performance
- **Window operations**: 700K+ ops/sec for basic operations
- **Compression throughput**: 13K-63K ops/sec depending on level
- **Strategy selection**: 50K-706K ops/sec depending on complexity
- **Overall system**: No performance regressions, several optimizations

## [0.1.0] - 2025-02-16

### Added
- Initial release of antaris-context
- **ContextManager** - Main class for managing context window budgets and optimization
- **ContextWindow** - Context window state tracking with per-section token management
- **MessageCompressor** - Content compression with configurable levels (light, moderate, aggressive)
- **ContextStrategy** framework with multiple selection strategies:
  - RecencyStrategy: Select newest content first
  - RelevanceStrategy: Select content based on query relevance
  - HybridStrategy: Combine recency and relevance with configurable weights
  - BudgetStrategy: Maximize value within token budget
- **ContextProfiler** - Analysis and optimization suggestions for context usage
- Token counting using whitespace-based approximation (~4 chars per token)
- Automatic truncation strategies (oldest-first, lowest-priority, smart-summary-markers)
- Priority-based content inclusion (critical > important > normal > optional)
- File-based JSON configuration support
- Comprehensive test suite with 40+ test cases
- Zero external dependencies (Python stdlib only)

### Features
- Context budget allocation across sections (system, memory, conversation, tools)
- Overflow detection and warnings
- Usage statistics and reporting
- Content compression with multiple strategies
- Pluggable content selection strategies
- Historical usage tracking and trend analysis
- Optimization suggestions based on usage patterns
- Export/import of context state (structure only)
- Budget reallocation suggestions
- Redundant content detection

### Design Principles
- Zero external dependencies (Python standard library only)
- File-based configuration using plain JSON
- Deterministic operations (no LLM API calls)
- Honest about limitations and approximations
- Apache 2.0 license for maximum compatibility

### Documentation
- Comprehensive README with real examples
- "What It Doesn't Do" section highlighting limitations
- Full API documentation in docstrings
- Configuration examples and best practices