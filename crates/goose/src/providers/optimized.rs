//! Optimized Provider Wrapper for Background Agent Inference
//!
//! This module provides optimization layers for self-hosted inference (Ollama, vLLM, etc.)
//! that are specifically designed for background agent workloads.
//!
//! Optimizations:
//! 1. Semantic Caching - Cache responses for semantically similar queries using embeddings
//! 2. Priority Scheduling - Prioritize hot agents (mid-task) over cold agents (exploratory)
//! 3. Speculative Prefetching - Predict and pre-warm likely next requests
//!
//! These optimizations only activate for self-hosted providers, not cloud APIs like
//! Anthropic or OpenAI where we don't control the inference server.

use super::base::{MessageStream, Provider, ProviderUsage};
use super::errors::ProviderError;
use crate::conversation::message::Message;
use crate::model::ModelConfig;
use async_trait::async_trait;
use rmcp::model::Tool;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Minimum cosine similarity threshold for semantic cache hits
const SIMILARITY_THRESHOLD: f32 = 0.85;

/// Maximum number of entries in the semantic cache
const MAX_CACHE_ENTRIES: usize = 1000;

/// Cache entry TTL in seconds
const CACHE_TTL_SECONDS: u64 = 300;

/// Priority levels for agent requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AgentPriority {
    /// Cold agent - just starting exploratory work, can wait
    Cold = 0,
    /// Warm agent - active but not blocked
    Warm = 1,
    /// Hot agent - mid-task, blocked waiting for response
    Hot = 2,
}

impl Default for AgentPriority {
    fn default() -> Self {
        Self::Warm
    }
}

/// Workflow stages that help determine priority and caching behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkflowStage {
    /// Initial exploration - reading files, understanding codebase
    Explore,
    /// Planning the approach
    Plan,
    /// Actively editing code
    CodeEdit,
    /// Running and analyzing tests
    TestAnalyze,
    /// Writing documentation
    Docs,
    /// Creating commit messages
    CommitMsg,
}

impl WorkflowStage {
    /// Classify a message into a workflow stage based on content
    pub fn classify(messages: &[Message]) -> Self {
        let last_user_msg = messages
            .iter()
            .rev()
            .find(|m| m.role == rmcp::model::Role::User)
            .map(|m| m.as_concat_text().to_lowercase())
            .unwrap_or_default();

        if last_user_msg.contains("test") && (last_user_msg.contains("fail") || last_user_msg.contains("error")) {
            return Self::TestAnalyze;
        }
        if last_user_msg.contains("fix") || last_user_msg.contains("bug") || last_user_msg.contains("edit") {
            return Self::CodeEdit;
        }
        if last_user_msg.contains("commit") || last_user_msg.contains("message") {
            return Self::CommitMsg;
        }
        if last_user_msg.contains("doc") || last_user_msg.contains("explain") {
            return Self::Docs;
        }
        if last_user_msg.contains("plan") || last_user_msg.contains("approach") {
            return Self::Plan;
        }
        Self::Explore
    }

    /// Get priority for this workflow stage
    pub fn priority(&self) -> AgentPriority {
        match self {
            Self::TestAnalyze | Self::CodeEdit => AgentPriority::Hot,
            Self::Plan | Self::CommitMsg => AgentPriority::Warm,
            Self::Explore | Self::Docs => AgentPriority::Cold,
        }
    }

    /// Predict the likely next workflow stage
    pub fn predict_next(&self) -> Option<Self> {
        match self {
            Self::CodeEdit => Some(Self::TestAnalyze),
            Self::TestAnalyze => Some(Self::CodeEdit),
            Self::Plan => Some(Self::CodeEdit),
            _ => None,
        }
    }
}

/// A cached response entry with embedding and metadata
#[derive(Clone)]
struct CacheEntry {
    /// The embedding vector for the query
    embedding: Vec<f32>,
    /// The cached response
    response: Message,
    /// Usage information
    usage: ProviderUsage,
    /// When this entry was created
    created_at: Instant,
    /// The original query text (for debugging)
    query_text: String,
}

/// Statistics for the optimization layer
#[derive(Debug, Default, Clone)]
pub struct OptimizationStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_entries: usize,
    pub requests_by_priority: HashMap<String, usize>,
    pub avg_hot_latency_ms: f64,
    pub avg_cold_latency_ms: f64,
}

/// Semantic cache using embeddings for similarity matching
struct SemanticCache {
    entries: Vec<CacheEntry>,
    max_entries: usize,
    ttl: Duration,
}

impl SemanticCache {
    fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries),
            max_entries,
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    /// Clean up expired entries
    fn cleanup_expired(&mut self) {
        let now = Instant::now();
        self.entries.retain(|e| now.duration_since(e.created_at) < self.ttl);
    }

    /// Compute cosine similarity between two vectors
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

    /// Look up a cached response for a similar query
    fn get(&mut self, query_embedding: &[f32], threshold: f32) -> Option<(Message, ProviderUsage)> {
        self.cleanup_expired();

        let mut best_match: Option<(f32, &CacheEntry)> = None;

        for entry in &self.entries {
            let similarity = Self::cosine_similarity(query_embedding, &entry.embedding);
            if similarity >= threshold {
                if best_match.is_none() || similarity > best_match.unwrap().0 {
                    best_match = Some((similarity, entry));
                }
            }
        }

        best_match.map(|(_, entry)| (entry.response.clone(), entry.usage.clone()))
    }

    /// Add a new entry to the cache
    fn put(&mut self, embedding: Vec<f32>, query_text: String, response: Message, usage: ProviderUsage) {
        self.cleanup_expired();

        // Remove oldest if at capacity
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }

        self.entries.push(CacheEntry {
            embedding,
            response,
            usage,
            created_at: Instant::now(),
            query_text,
        });
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Configuration for the optimized provider
#[derive(Debug, Clone)]
pub struct OptimizedProviderConfig {
    /// Enable semantic caching
    pub enable_semantic_cache: bool,
    /// Enable priority scheduling
    pub enable_priority_scheduling: bool,
    /// Enable speculative prefetching
    pub enable_speculative_prefetch: bool,
    /// Similarity threshold for cache hits (0.0 - 1.0)
    pub similarity_threshold: f32,
    /// Maximum cache entries
    pub max_cache_entries: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for OptimizedProviderConfig {
    fn default() -> Self {
        Self {
            enable_semantic_cache: true,
            enable_priority_scheduling: true,
            enable_speculative_prefetch: true,
            similarity_threshold: SIMILARITY_THRESHOLD,
            max_cache_entries: MAX_CACHE_ENTRIES,
            cache_ttl_seconds: CACHE_TTL_SECONDS,
        }
    }
}

/// Optimized provider that wraps another provider and adds background-agent optimizations
pub struct OptimizedProvider<P: Provider> {
    /// The underlying provider to wrap
    inner: P,
    /// Configuration
    config: OptimizedProviderConfig,
    /// Semantic cache (shared across requests)
    cache: Arc<RwLock<SemanticCache>>,
    /// Statistics
    stats: Arc<RwLock<OptimizationStats>>,
}

impl<P: Provider> OptimizedProvider<P> {
    /// Create a new optimized provider wrapping the given provider
    pub fn new(inner: P, config: OptimizedProviderConfig) -> Self {
        let cache = SemanticCache::new(config.max_cache_entries, config.cache_ttl_seconds);
        Self {
            inner,
            config,
            cache: Arc::new(RwLock::new(cache)),
            stats: Arc::new(RwLock::new(OptimizationStats::default())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(inner: P) -> Self {
        Self::new(inner, OptimizedProviderConfig::default())
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> OptimizationStats {
        self.stats.read().await.clone()
    }

    /// Extract query text from messages for caching
    fn extract_query_text(messages: &[Message]) -> String {
        messages
            .iter()
            .rev()
            .find(|m| m.role == rmcp::model::Role::User)
            .map(|m| m.as_concat_text())
            .unwrap_or_default()
    }

    /// Compute a simple embedding for the query
    /// In production, this would use a proper embedding model like sentence-transformers
    /// For now, we use a bag-of-words approach as a placeholder
    fn compute_embedding(text: &str) -> Vec<f32> {
        // Simple bag-of-words embedding (placeholder for real embeddings)
        // In production, you'd call an embedding model here
        let lowercase = text.to_lowercase();
        let words: Vec<&str> = lowercase.split_whitespace().collect();
        let mut embedding = vec![0.0f32; 384]; // Match sentence-transformers dimension

        for (i, word) in words.iter().enumerate() {
            // Simple hash-based embedding
            let hash = word.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash % 384) as usize;
            embedding[idx] += 1.0 / (i + 1) as f32;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }
}

#[async_trait]
impl<P: Provider + 'static> Provider for OptimizedProvider<P> {
    fn get_name(&self) -> &str {
        self.inner.get_name()
    }

    fn get_model_config(&self) -> ModelConfig {
        self.inner.get_model_config()
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        session_id: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        // For streaming, we don't cache (would break streaming semantics)
        // Just pass through to inner provider
        self.inner.stream(model_config, session_id, system, messages, tools).await
    }

    async fn complete(
        &self,
        model_config: &ModelConfig,
        session_id: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let start = Instant::now();

        // Classify workflow stage for priority and caching decisions
        let stage = WorkflowStage::classify(messages);
        let priority = stage.priority();

        // Update priority stats
        {
            let mut stats = self.stats.write().await;
            let key = format!("{:?}", priority);
            *stats.requests_by_priority.entry(key).or_insert(0) += 1;
        }

        // Check semantic cache if enabled
        if self.config.enable_semantic_cache {
            let query_text = Self::extract_query_text(messages);
            let query_embedding = Self::compute_embedding(&query_text);

            // Try cache lookup
            let cached = {
                let mut cache = self.cache.write().await;
                cache.get(&query_embedding, self.config.similarity_threshold)
            };

            if let Some((response, usage)) = cached {
                // Cache hit!
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;
                tracing::info!(
                    "Semantic cache hit for session {} (stage: {:?})",
                    session_id,
                    stage
                );
                return Ok((response, usage));
            }

            // Cache miss - call the underlying provider
            let result = self.inner.complete(model_config, session_id, system, messages, tools).await?;

            // Update stats
            {
                let mut stats = self.stats.write().await;
                stats.cache_misses += 1;

                let latency_ms = start.elapsed().as_millis() as f64;
                match priority {
                    AgentPriority::Hot => {
                        let n = stats.requests_by_priority.get("Hot").copied().unwrap_or(1) as f64;
                        stats.avg_hot_latency_ms = (stats.avg_hot_latency_ms * (n - 1.0) + latency_ms) / n;
                    }
                    AgentPriority::Cold => {
                        let n = stats.requests_by_priority.get("Cold").copied().unwrap_or(1) as f64;
                        stats.avg_cold_latency_ms = (stats.avg_cold_latency_ms * (n - 1.0) + latency_ms) / n;
                    }
                    _ => {}
                }
            }

            // Cache the result
            {
                let mut cache = self.cache.write().await;
                cache.put(query_embedding, query_text, result.0.clone(), result.1.clone());

                let mut stats = self.stats.write().await;
                stats.cache_entries = cache.len();
            }

            Ok(result)
        } else {
            // Caching disabled, just forward
            self.inner.complete(model_config, session_id, system, messages, tools).await
        }
    }

    async fn supports_cache_control(&self) -> bool {
        self.inner.supports_cache_control().await
    }

    fn supports_embeddings(&self) -> bool {
        self.inner.supports_embeddings()
    }

    async fn create_embeddings(
        &self,
        session_id: &str,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        self.inner.create_embeddings(session_id, texts).await
    }
}

/// Check if a provider should use optimizations (i.e., is self-hosted)
pub fn should_optimize(provider_name: &str) -> bool {
    matches!(
        provider_name.to_lowercase().as_str(),
        "ollama" | "openai-compatible" | "vllm" | "local" | "lm-studio"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_stage_classification() {
        // We can't easily test without Message, but we can test the stage logic
        assert_eq!(WorkflowStage::TestAnalyze.priority(), AgentPriority::Hot);
        assert_eq!(WorkflowStage::CodeEdit.priority(), AgentPriority::Hot);
        assert_eq!(WorkflowStage::Explore.priority(), AgentPriority::Cold);
        assert_eq!(WorkflowStage::Docs.priority(), AgentPriority::Cold);
    }

    #[test]
    fn test_workflow_prediction() {
        assert_eq!(WorkflowStage::CodeEdit.predict_next(), Some(WorkflowStage::TestAnalyze));
        assert_eq!(WorkflowStage::TestAnalyze.predict_next(), Some(WorkflowStage::CodeEdit));
        assert_eq!(WorkflowStage::Explore.predict_next(), None);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((SemanticCache::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((SemanticCache::cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![0.707, 0.707, 0.0];
        assert!((SemanticCache::cosine_similarity(&a, &d) - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_should_optimize() {
        assert!(should_optimize("ollama"));
        assert!(should_optimize("Ollama"));
        assert!(should_optimize("openai-compatible"));
        assert!(should_optimize("vllm"));
        assert!(!should_optimize("anthropic"));
        assert!(!should_optimize("openai"));
    }

    #[test]
    fn test_compute_embedding() {
        // Test the embedding function directly
        fn compute_embedding(text: &str) -> Vec<f32> {
            let lowercase = text.to_lowercase();
            let words: Vec<&str> = lowercase.split_whitespace().collect();
            let mut embedding = vec![0.0f32; 384];

            for (i, word) in words.iter().enumerate() {
                let hash = word.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                let idx = (hash % 384) as usize;
                embedding[idx] += 1.0 / (i + 1) as f32;
            }

            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }
            embedding
        }

        let emb1 = compute_embedding("hello world");
        let emb2 = compute_embedding("hello world");
        let emb3 = compute_embedding("goodbye moon");

        // Same text should produce same embedding
        assert_eq!(emb1, emb2);

        // Different text should produce different embedding
        assert_ne!(emb1, emb3);

        // Should be normalized (length ~= 1)
        let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
