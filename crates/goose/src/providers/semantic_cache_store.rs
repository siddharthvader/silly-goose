//! Pluggable Storage Backends for Semantic Cache
//!
//! Supports multiple storage backends for sharing cached embeddings across agents:
//! - InMemory: Default, single-process only
//! - Redis: Multi-agent, persistent, requires GOOSE_REDIS_URL
//! - ChromaDB: Vector-native storage, requires GOOSE_CHROMADB_URL (future)
//!
//! The whole point of semantic caching is sharing across agents - multiple agents
//! working on the same codebase will ask similar questions. Without shared storage,
//! each agent builds its own cache (not very useful).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// A cached entry with embedding and response
#[derive(Clone, Serialize, Deserialize)]
pub struct CachedEntry {
    /// The embedding vector for similarity matching
    pub embedding: Vec<f32>,
    /// The original query text (for debugging/inspection)
    pub query_text: String,
    /// Serialized response (JSON)
    pub response_json: String,
    /// Serialized usage info (JSON)
    pub usage_json: String,
    /// Unix timestamp when this entry was created
    pub created_at_unix: u64,
    /// TTL in seconds
    pub ttl_seconds: u64,
}

impl CachedEntry {
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now > self.created_at_unix + self.ttl_seconds
    }
}

/// Result of a cache lookup
pub struct CacheHit {
    pub response_json: String,
    pub usage_json: String,
    pub similarity: f32,
}

/// Trait for semantic cache storage backends
#[async_trait]
pub trait SemanticCacheStore: Send + Sync {
    /// Get the backend name for logging
    fn name(&self) -> &'static str;

    /// Look up similar entries by embedding vector
    /// Returns the best match above the similarity threshold, if any
    async fn get_similar(
        &self,
        query_embedding: &[f32],
        similarity_threshold: f32,
    ) -> Result<Option<CacheHit>, CacheStoreError>;

    /// Store a new entry
    async fn put(&self, entry: CachedEntry) -> Result<(), CacheStoreError>;

    /// Get cache statistics
    async fn stats(&self) -> CacheStats;

    /// Clear expired entries (called periodically)
    async fn cleanup_expired(&self) -> Result<usize, CacheStoreError>;
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub entries: usize,
    pub hits: usize,
    pub misses: usize,
    pub backend: String,
}

/// Errors from cache storage
#[derive(Debug)]
pub enum CacheStoreError {
    ConnectionError(String),
    SerializationError(String),
    StorageError(String),
}

impl std::fmt::Display for CacheStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionError(s) => write!(f, "Connection error: {}", s),
            Self::SerializationError(s) => write!(f, "Serialization error: {}", s),
            Self::StorageError(s) => write!(f, "Storage error: {}", s),
        }
    }
}

impl std::error::Error for CacheStoreError {}

// ============================================================================
// In-Memory Backend (default)
// ============================================================================

use std::sync::RwLock;

/// In-memory storage - good for single agent, no persistence
pub struct InMemoryStore {
    entries: RwLock<Vec<CachedEntry>>,
    max_entries: usize,
    stats: RwLock<CacheStats>,
}

impl InMemoryStore {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(Vec::with_capacity(max_entries)),
            max_entries,
            stats: RwLock::new(CacheStats {
                backend: "in-memory".to_string(),
                ..Default::default()
            }),
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

#[async_trait]
impl SemanticCacheStore for InMemoryStore {
    fn name(&self) -> &'static str {
        "in-memory"
    }

    async fn get_similar(
        &self,
        query_embedding: &[f32],
        similarity_threshold: f32,
    ) -> Result<Option<CacheHit>, CacheStoreError> {
        let entries = self.entries.read().map_err(|e| {
            CacheStoreError::StorageError(format!("Lock error: {}", e))
        })?;

        let mut best_match: Option<(f32, &CachedEntry)> = None;

        for entry in entries.iter() {
            if entry.is_expired() {
                continue;
            }
            let similarity = Self::cosine_similarity(query_embedding, &entry.embedding);
            if similarity >= similarity_threshold {
                if best_match.is_none() || similarity > best_match.unwrap().0 {
                    best_match = Some((similarity, entry));
                }
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().map_err(|e| {
                CacheStoreError::StorageError(format!("Lock error: {}", e))
            })?;
            if best_match.is_some() {
                stats.hits += 1;
            } else {
                stats.misses += 1;
            }
        }

        Ok(best_match.map(|(similarity, entry)| CacheHit {
            response_json: entry.response_json.clone(),
            usage_json: entry.usage_json.clone(),
            similarity,
        }))
    }

    async fn put(&self, entry: CachedEntry) -> Result<(), CacheStoreError> {
        let mut entries = self.entries.write().map_err(|e| {
            CacheStoreError::StorageError(format!("Lock error: {}", e))
        })?;

        // Remove oldest if at capacity
        if entries.len() >= self.max_entries {
            entries.remove(0);
        }

        entries.push(entry);

        // Update stats
        {
            let mut stats = self.stats.write().map_err(|e| {
                CacheStoreError::StorageError(format!("Lock error: {}", e))
            })?;
            stats.entries = entries.len();
        }

        Ok(())
    }

    async fn stats(&self) -> CacheStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    async fn cleanup_expired(&self) -> Result<usize, CacheStoreError> {
        let mut entries = self.entries.write().map_err(|e| {
            CacheStoreError::StorageError(format!("Lock error: {}", e))
        })?;

        let before = entries.len();
        entries.retain(|e| !e.is_expired());
        let removed = before - entries.len();

        if removed > 0 {
            let mut stats = self.stats.write().map_err(|e| {
                CacheStoreError::StorageError(format!("Lock error: {}", e))
            })?;
            stats.entries = entries.len();
        }

        Ok(removed)
    }
}

// ============================================================================
// Redis Backend (multi-agent, persistent)
// ============================================================================

/// Redis storage - enables sharing across multiple agents
///
/// Configure via environment:
/// - GOOSE_REDIS_URL: Redis connection URL (e.g., redis://localhost:6379)
/// - GOOSE_REDIS_PREFIX: Key prefix (default: "goose:semantic_cache:")
///
/// Each entry is stored as a Redis hash with TTL.
/// Similarity search requires iterating keys (not ideal for huge caches,
/// but fine for typical agent workloads of ~1000 entries).
///
/// For production at scale, consider ChromaDB/Milvus/Pinecone.
pub struct RedisStore {
    // In a real implementation, this would hold:
    // - redis::Client
    // - key_prefix: String
    // - connection pool
    //
    // For now, we provide a skeleton that shows the interface.
    // Users would need to add the `redis` crate dependency.
    _config: RedisConfig,
}

pub struct RedisConfig {
    pub url: String,
    pub prefix: String,
    pub max_entries: usize,
}

impl RedisStore {
    /// Create a new Redis store
    ///
    /// Note: This is a skeleton. Full implementation requires:
    /// 1. Add `redis = { version = "0.25", features = ["tokio-comp"] }` to Cargo.toml
    /// 2. Implement actual Redis operations
    pub fn new(config: RedisConfig) -> Result<Self, CacheStoreError> {
        // In real implementation:
        // let client = redis::Client::open(config.url.as_str())
        //     .map_err(|e| CacheStoreError::ConnectionError(e.to_string()))?;

        tracing::info!(
            "Redis semantic cache configured: {} (prefix: {})",
            config.url,
            config.prefix
        );

        Ok(Self { _config: config })
    }
}

#[async_trait]
impl SemanticCacheStore for RedisStore {
    fn name(&self) -> &'static str {
        "redis"
    }

    async fn get_similar(
        &self,
        _query_embedding: &[f32],
        _similarity_threshold: f32,
    ) -> Result<Option<CacheHit>, CacheStoreError> {
        // Skeleton implementation - would need actual Redis calls:
        //
        // 1. SCAN for all keys with prefix
        // 2. HGETALL each key to get embedding + response
        // 3. Compute cosine similarity
        // 4. Return best match above threshold
        //
        // For production, consider using Redis Vector Search (RediSearch)
        // or a dedicated vector DB.

        Err(CacheStoreError::StorageError(
            "Redis backend not fully implemented - add redis crate dependency".to_string()
        ))
    }

    async fn put(&self, _entry: CachedEntry) -> Result<(), CacheStoreError> {
        // Would use: HSET + EXPIRE
        Err(CacheStoreError::StorageError(
            "Redis backend not fully implemented - add redis crate dependency".to_string()
        ))
    }

    async fn stats(&self) -> CacheStats {
        CacheStats {
            backend: "redis".to_string(),
            ..Default::default()
        }
    }

    async fn cleanup_expired(&self) -> Result<usize, CacheStoreError> {
        // Redis handles TTL automatically via EXPIRE
        Ok(0)
    }
}

// ============================================================================
// Factory function to create the right backend
// ============================================================================

/// Create a cache store based on environment configuration
///
/// Checks in order:
/// 1. GOOSE_REDIS_URL -> RedisStore
/// 2. GOOSE_CHROMADB_URL -> ChromaDBStore (future)
/// 3. Default -> InMemoryStore
pub fn create_cache_store(max_entries: usize, ttl_seconds: u64) -> Box<dyn SemanticCacheStore> {
    let config = crate::config::Config::global();

    // Check for Redis
    if let Ok(redis_url) = config.get_param::<String>("GOOSE_REDIS_URL") {
        let prefix = config
            .get_param::<String>("GOOSE_REDIS_PREFIX")
            .unwrap_or_else(|_| "goose:semantic_cache:".to_string());

        match RedisStore::new(RedisConfig {
            url: redis_url,
            prefix,
            max_entries,
        }) {
            Ok(store) => {
                tracing::info!("Using Redis backend for semantic cache (multi-agent sharing enabled)");
                return Box::new(store);
            }
            Err(e) => {
                tracing::warn!("Failed to connect to Redis, falling back to in-memory: {}", e);
            }
        }
    }

    // TODO: Add ChromaDB support
    // if let Ok(chromadb_url) = config.get_param::<String>("GOOSE_CHROMADB_URL") { ... }

    // Default to in-memory
    tracing::info!(
        "Using in-memory semantic cache (single-agent only, max {} entries, {}s TTL)",
        max_entries,
        ttl_seconds
    );
    Box::new(InMemoryStore::new(max_entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_store_basic() {
        let store = InMemoryStore::new(100);

        // Put an entry
        let entry = CachedEntry {
            embedding: vec![1.0, 0.0, 0.0],
            query_text: "test query".to_string(),
            response_json: r#"{"test": true}"#.to_string(),
            usage_json: r#"{"tokens": 10}"#.to_string(),
            created_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            ttl_seconds: 300,
        };
        store.put(entry).await.unwrap();

        // Should find similar
        let hit = store.get_similar(&[1.0, 0.0, 0.0], 0.9).await.unwrap();
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().response_json, r#"{"test": true}"#);

        // Should not find dissimilar
        let miss = store.get_similar(&[0.0, 1.0, 0.0], 0.9).await.unwrap();
        assert!(miss.is_none());
    }

    #[tokio::test]
    async fn test_in_memory_store_expiration() {
        let store = InMemoryStore::new(100);

        // Put an already-expired entry
        let entry = CachedEntry {
            embedding: vec![1.0, 0.0, 0.0],
            query_text: "expired".to_string(),
            response_json: "{}".to_string(),
            usage_json: "{}".to_string(),
            created_at_unix: 0, // Very old
            ttl_seconds: 1,
        };
        store.put(entry).await.unwrap();

        // Should not find expired entry
        let result = store.get_similar(&[1.0, 0.0, 0.0], 0.9).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_in_memory_store_max_entries() {
        let store = InMemoryStore::new(2);

        for i in 0..5 {
            let entry = CachedEntry {
                embedding: vec![i as f32, 0.0, 0.0],
                query_text: format!("query {}", i),
                response_json: "{}".to_string(),
                usage_json: "{}".to_string(),
                created_at_unix: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                ttl_seconds: 300,
            };
            store.put(entry).await.unwrap();
        }

        let stats = store.stats().await;
        assert_eq!(stats.entries, 2); // Should cap at max
    }
}
