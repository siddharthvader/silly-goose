<div align="center">

# silly-goose 🪿

_Fork of [block/goose](https://github.com/block/goose) with inference optimizations for background agents_

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"
    ><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>
</div>

## Overview

This fork adds **inference optimizations for background agent workloads** when using self-hosted models (Ollama, vLLM, etc.). These optimizations exploit the unique properties of background agents—predictable workflows, tolerance for batching, and multi-agent coordination—to reduce latency and inference costs.

| Optimization | Speedup | Description |
|-------------|---------|-------------|
| **Semantic Caching** | ~3000x per hit | Cache responses for semantically similar queries |
| **Priority Scheduling** | 1.4x | Serve blocked agents before exploratory ones |
| **Speculative Prefetching** | 2.0x | Predict and prefetch likely next requests |

## Why Background Agents Are Different

| Property | Interactive Chat | Background Agents |
|----------|-----------------|-------------------|
| User waiting? | Yes | No |
| Predictable workflow? | No | Yes (edit → test → fix) |
| Multiple instances? | Usually 1 | Often many |
| Can batch/delay? | No | Yes |

## Quick Start

```bash
# Clone
git clone https://github.com/siddharthvader/silly-goose.git
cd silly-goose

# Build
cargo build -p goose --release

# Enable optimizations (only affects self-hosted providers)
export GOOSE_ENABLE_OPTIMIZATIONS=true

# Optional: Redis for multi-agent cache sharing
export GOOSE_REDIS_URL=redis://localhost:6379
```

## Configuration

### Enable Optimizations

```bash
# Master switch - only affects self-hosted providers (Ollama, vLLM, etc.)
export GOOSE_ENABLE_OPTIMIZATIONS=true
```

Cloud providers (Anthropic, OpenAI, Google) are **not affected**—they manage their own inference.

### Semantic Cache

```bash
# Redis for multi-agent cache sharing
export GOOSE_REDIS_URL=redis://localhost:6379

# Tuning (optional)
export GOOSE_SEMANTIC_CACHE_THRESHOLD=0.85  # Similarity threshold
export GOOSE_SEMANTIC_CACHE_MAX_ENTRIES=1000
export GOOSE_SEMANTIC_CACHE_TTL=300  # Seconds
```

### Individual Toggles

```bash
export GOOSE_ENABLE_SEMANTIC_CACHE=true       # Default: true
export GOOSE_ENABLE_PRIORITY_SCHEDULING=true  # Default: true
export GOOSE_ENABLE_SPECULATIVE_PREFETCH=true # Default: true
```

## How It Works

### Semantic Caching

Multiple agents asking similar questions get cached responses:

```
Agent 1: "What does authenticate() do?"      → LLM call → cache
Agent 2: "Explain the authenticate function" → cache hit (<1ms)
Agent 3: "How does authenticate work?"       → cache hit (<1ms)
```

Uses sentence-transformer embeddings (`all-MiniLM-L6-v2`) with cosine similarity matching (threshold: 0.85).

### Priority Scheduling

Requests are classified by workflow stage:

| Stage | Priority | Example |
|-------|----------|---------|
| TestAnalyze | Hot | "Test failed with AssertionError..." |
| CodeEdit | Hot | "Fix this bug..." |
| Plan | Warm | "How should I approach..." |
| Explore | Cold | "What does this file do?" |

Hot agents (blocked, waiting) get served before cold agents (exploring).

### Speculative Prefetching

Predicts likely next requests based on workflow patterns:

- After `CodeEdit` → prefetch `TestAnalyze`
- After `TestAnalyze` → prefetch `CodeEdit`
- After `Plan` → prefetch `CodeEdit`

## Project Structure

```
crates/goose/src/providers/
├── optimized.rs            # Optimization wrapper
├── semantic_cache_store.rs # Pluggable storage backends (InMemory, Redis)
└── mod.rs                  # Module exports

OPTIMIZATIONS.pdf           # Technical documentation
```

## Benchmarks

Tested with Qwen2.5-Coder-7B via vLLM on A100:

| Optimization | Result | Notes |
|-------------|--------|-------|
| Semantic Caching | **~3000x** per cache hit | Embeddings + cosine similarity |
| Priority Scheduling | **1.4x** | Hot-first sequencing |
| Speculative Prefetch | **2.0x** | Parallel execution |

Optimizations that didn't help:
- Context compression (0.89x) — generation time dominates
- Accumulation window (1.01x) — vLLM already batches internally

## Upstream

Fork of [block/goose](https://github.com/block/goose). All original features work as expected. Optimizations are opt-in and only affect self-hosted inference.

```bash
# Sync with upstream
git remote add upstream https://github.com/block/goose.git
git fetch upstream
git merge upstream/main
```

## License

Apache 2.0 (same as upstream)
