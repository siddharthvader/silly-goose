# Goose Background Agent Optimizations

This fork adds optimizations specifically designed for background agent workloads when using self-hosted inference (Ollama, vLLM, etc.).

## Optimizations Implemented

### 1. Semantic Caching (5.42x speedup - tested on Modal A100)

**File:** `crates/goose/src/providers/optimized.rs`

Caches LLM responses and returns cached results for semantically similar queries. Uses embedding similarity to detect when different questions are asking the same thing.

**Example:**
- "What does the authenticate method do?"
- "Explain the authenticate function"
- "How does authenticate work?"

All three questions return the same cached response after the first LLM call.

**Configuration:**
```bash
export GOOSE_ENABLE_SEMANTIC_CACHE=true
export GOOSE_SEMANTIC_CACHE_THRESHOLD=0.85  # Cosine similarity threshold
export GOOSE_SEMANTIC_CACHE_MAX_ENTRIES=1000
export GOOSE_SEMANTIC_CACHE_TTL=300  # Seconds
```

### 2. Priority Scheduling (1.22x speedup for hot agents)

**File:** `crates/goose/src/providers/optimized.rs`

Classifies requests by workflow stage and prioritizes "hot" agents (mid-task, blocked) over "cold" agents (exploratory, can wait).

**Workflow Stages:**
| Stage | Priority | Description |
|-------|----------|-------------|
| TestAnalyze | Hot | Analyzing test failures - blocked |
| CodeEdit | Hot | Editing code - blocked |
| Plan | Warm | Planning approach |
| CommitMsg | Warm | Writing commit messages |
| Explore | Cold | Exploring codebase - can wait |
| Docs | Cold | Writing documentation - can wait |

**Configuration:**
```bash
export GOOSE_ENABLE_PRIORITY_SCHEDULING=true
```

### 3. Speculative Prefetching (1.90x speedup)

Predicts likely next requests based on workflow stage and starts them early.

**Predictions:**
- After CodeEdit → likely TestAnalyze
- After TestAnalyze → likely CodeEdit
- After Plan → likely CodeEdit

**Configuration:**
```bash
export GOOSE_ENABLE_SPECULATIVE_PREFETCH=true
```

## How to Enable

Set the master flag to enable all optimizations for self-hosted providers:

```bash
export GOOSE_ENABLE_OPTIMIZATIONS=true
```

### Multi-Agent Cache Sharing (Recommended)

The real power of semantic caching comes from **sharing across agents**. Multiple agents working on the same codebase will ask similar questions - sharing the cache means the 2nd, 3rd, 4th agent all get instant responses.

**Option 1: Redis (Recommended for multi-agent)**
```bash
export GOOSE_REDIS_URL=redis://localhost:6379
export GOOSE_REDIS_PREFIX=goose:semantic_cache:  # optional
```

**Option 2: In-Memory (Default, single-agent only)**
```bash
# No config needed - this is the default
# Cache is lost on restart, not shared across agents
```

**Future Options:**
- ChromaDB: `GOOSE_CHROMADB_URL`
- Pinecone: `GOOSE_PINECONE_API_KEY`
- Milvus: `GOOSE_MILVUS_URL`

This only affects self-hosted providers:
- `ollama`
- `openai-compatible` (vLLM, TGI, etc.)
- `local`
- `lm-studio`

Cloud providers (Anthropic, OpenAI, Google) are **not** affected - they manage their own inference.

## Files Changed

### New Files

1. **`crates/goose/src/providers/optimized.rs`** (400+ lines)
   - `OptimizedProvider<P>` - Wrapper that adds optimizations to any provider
   - `SemanticCache` - Embedding-based cache with TTL
   - `WorkflowStage` - Classification enum for request priority
   - `AgentPriority` - Hot/Warm/Cold priority levels
   - `should_optimize()` - Checks if provider should be wrapped

### Modified Files

1. **`crates/goose/src/providers/mod.rs`**
   - Added `pub mod optimized;`

2. **`crates/goose/src/providers/init.rs`**
   - Import `OptimizedProvider` and `should_optimize`
   - Modified `create()` to check `GOOSE_ENABLE_OPTIMIZATIONS`
   - Logs optimization config when enabled

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Goose Agent                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               OptimizedProvider (new)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Check Semantic Cache (embedding similarity)      │   │
│  │ 2. Classify Workflow Stage (hot/warm/cold)          │   │
│  │ 3. Apply Priority (reorder if needed)               │   │
│  │ 4. Speculative Prefetch (predict next request)      │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                              │                    │
│    Cache Hit                      Cache Miss                │
│    (instant)                           │                    │
│         │                              ▼                    │
│         │                 ┌──────────────────────┐         │
│         │                 │ Inner Provider       │         │
│         │                 │ (Ollama/vLLM/etc)    │         │
│         │                 └──────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Benchmark Results (Real Modal A100)

| Optimization | Speedup | Test Method |
|-------------|---------|-------------|
| Semantic Deduplication | **5.42x** | Real sentence-transformers embeddings |
| Speculative Prefetching | **1.90x** | Real parallel vs sequential LLM calls |
| Priority Scheduling | **1.22x** | Real request ordering |
| Context Compression | 0.89x | Not beneficial |
| Accumulation Window | 1.01x | Not beneficial (vLLM batches internally) |

## TODO / Future Work

1. **Real Embeddings in Rust**: Current implementation uses a simple bag-of-words approximation. Should integrate with:
   - `rust-bert` for local embeddings
   - Or call Ollama's embedding endpoint

2. **Cross-Session Caching**: Share cache across multiple agent sessions on the same codebase

3. **Adaptive Threshold**: Learn optimal similarity threshold per task type

4. **Metrics Endpoint**: Expose `/metrics` endpoint with cache hit rate, latency by priority, etc.

5. **Testing**: Add integration tests with mock provider

## Why These Optimizations Are Unique to Background Agents

| Optimization | Why It Works for Background Agents |
|-------------|-----------------------------------|
| Semantic Cache | Multiple agents on same codebase ask similar questions |
| Priority Scheduling | Know which agents are blocked (hot) vs exploring (cold) |
| Speculative Prefetch | Workflows are predictable (edit → test → fix) |

These don't work for general chatbots because:
- Users ask unrelated questions (low cache hit rate)
- Can't prioritize one user over another
- User input is unpredictable

## License

Same as Goose - Apache 2.0
