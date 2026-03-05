<div align="center">

# silly-goose 🪿

_Fork of [block/goose](https://github.com/block/goose) with background-agent inference optimizations_

> Why "silly-goose"? Because optimizing inference for background agents is a mass of fun.

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"
    ><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>
</div>

## What's Different?

This fork adds **inference optimizations specifically designed for background agent workloads** when using self-hosted models (Ollama, vLLM, etc.).

| Optimization | Speedup | Description |
|-------------|---------|-------------|
| **Semantic Caching** | 5.42x | Cache responses for similar queries using embeddings |
| **Priority Scheduling** | 1.22x | Prioritize blocked agents over exploratory work |
| **Speculative Prefetching** | 1.90x | Predict and pre-warm likely next requests |

All speedups verified on **real Modal A100 GPU** - not mocked.

## Why These Optimizations?

Background agents are different from interactive chat:

| Property | Interactive Chat | Background Agents |
|----------|-----------------|-------------------|
| User waiting? | Yes | No |
| Predictable workflow? | No | Yes (edit → test → fix) |
| Multiple instances? | Usually 1 | Often many |
| Can delay requests? | No | Yes |

This fork exploits these differences to reduce inference costs and latency.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/goose-optimized.git
cd goose-optimized

# Build
cargo build -p goose

# Enable optimizations (only for self-hosted providers)
export GOOSE_ENABLE_OPTIMIZATIONS=true

# For multi-agent cache sharing (recommended)
export GOOSE_REDIS_URL=redis://localhost:6379
```

## Configuration

### Enable Optimizations

```bash
# Master switch - only affects self-hosted providers (Ollama, vLLM, etc.)
export GOOSE_ENABLE_OPTIMIZATIONS=true
```

Cloud providers (Anthropic, OpenAI, Google) are **not affected** - they manage their own inference.

### Semantic Cache

Share cached responses across multiple agents:

```bash
# Use Redis for multi-agent sharing (recommended)
export GOOSE_REDIS_URL=redis://localhost:6379

# Tuning (optional)
export GOOSE_SEMANTIC_CACHE_THRESHOLD=0.85  # Similarity threshold
export GOOSE_SEMANTIC_CACHE_MAX_ENTRIES=1000
export GOOSE_SEMANTIC_CACHE_TTL=300  # Seconds
```

### Individual Toggles

```bash
export GOOSE_ENABLE_SEMANTIC_CACHE=true      # Default: true
export GOOSE_ENABLE_PRIORITY_SCHEDULING=true  # Default: true
export GOOSE_ENABLE_SPECULATIVE_PREFETCH=true # Default: true
```

## How It Works

### Semantic Caching (5.42x)

Multiple agents asking similar questions get cached responses:

```
Agent 1: "What does authenticate() do?"     → LLM call (5s) → cache
Agent 2: "Explain the authenticate function" → cache hit (<1ms)
Agent 3: "How does authenticate work?"       → cache hit (<1ms)
```

Uses sentence-transformer embeddings with cosine similarity matching.

### Priority Scheduling (1.22x)

Requests are classified by workflow stage:

| Stage | Priority | Example |
|-------|----------|---------|
| TestAnalyze | Hot | "Test failed with AssertionError..." |
| CodeEdit | Hot | "Fix this bug..." |
| Plan | Warm | "How should I approach..." |
| Explore | Cold | "What does this file do?" |

Hot agents (blocked, waiting) get served before cold agents (just exploring).

### Speculative Prefetching (1.90x)

Predicts likely next requests based on workflow:

- After `CodeEdit` → prefetch `TestAnalyze`
- After `TestAnalyze` → prefetch `CodeEdit`
- After `Plan` → prefetch `CodeEdit`

## Files Added

```
crates/goose/src/providers/
├── optimized.rs           # Main optimization wrapper
├── semantic_cache_store.rs # Pluggable storage backends
└── mod.rs                  # Updated to include new modules

OPTIMIZATION_README.md      # Detailed documentation
```

## Benchmark Results

Tested on Modal A100 with Qwen2.5-Coder-32B-Instruct via vLLM:

| Test | Result | Notes |
|------|--------|-------|
| Semantic Deduplication | **5.42x** | Real embeddings, 80% hit rate |
| Speculative Prefetching | **1.90x** | Parallel vs sequential |
| Priority Scheduling | **1.22x** | Hot agents faster |
| Context Compression | 0.89x | Not beneficial |
| Accumulation Window | 1.01x | vLLM already batches |

## Upstream

This is a fork of [block/goose](https://github.com/block/goose). All original features work as expected. Optimizations are opt-in and only affect self-hosted inference.

To sync with upstream:
```bash
git remote add upstream https://github.com/block/goose.git
git fetch upstream
git merge upstream/main
```

## License

Apache 2.0 (same as upstream)
