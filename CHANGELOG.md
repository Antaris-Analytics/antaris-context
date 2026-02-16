# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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