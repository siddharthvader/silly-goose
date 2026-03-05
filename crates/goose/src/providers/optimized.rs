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
use crate::conversation::message::{Message, MessageContent};
use rmcp::model::TextContent;
use crate::model::ModelConfig;
use async_trait::async_trait;
use chrono::Utc;
use rmcp::model::{RawTextContent, Role, Tool};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};

/// Minimum cosine similarity threshold for semantic cache hits
const SIMILARITY_THRESHOLD: f32 = 0.85;

/// Maximum number of entries in the semantic cache
const MAX_CACHE_ENTRIES: usize = 1000;

/// Cache entry TTL in seconds
const CACHE_TTL_SECONDS: u64 = 300;

/// Maximum concurrent hot requests (priority scheduling)
const MAX_CONCURRENT_HOT: usize = 4;

/// Maximum concurrent cold requests
const MAX_CONCURRENT_COLD: usize = 2;

/// Priority levels for agent requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
        if last_user_msg.contains("commit") {
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

    /// Predict the likely next workflow stage (for speculative prefetching)
    pub fn predict_next(&self) -> Option<Self> {
        match self {
            Self::CodeEdit => Some(Self::TestAnalyze),
            Self::TestAnalyze => Some(Self::CodeEdit),
            Self::Plan => Some(Self::CodeEdit),
            _ => None,
        }
    }

    /// Generate a prefetch query for the predicted next stage
    pub fn generate_prefetch_query(&self, context: &str) -> Option<String> {
        match self.predict_next()? {
            Self::TestAnalyze => Some(format!(
                "Based on the code changes, what tests might fail and why?\n\nContext: {}",
                &context[..context.len().min(500)]
            )),
            Self::CodeEdit => Some(format!(
                "What code changes are needed to fix the issues?\n\nContext: {}",
                &context[..context.len().min(500)]
            )),
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
    #[allow(dead_code)]
    query_text: String,
}

/// Statistics for the optimization layer
#[derive(Debug, Default, Clone)]
pub struct OptimizationStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_entries: usize,
    pub requests_by_priority: HashMap<String, usize>,
    pub prefetch_hits: usize,
    pub prefetch_misses: usize,
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
    fn get(&mut self, query_embedding: &[f32], threshold: f32) -> Option<(Message, ProviderUsage, f32)> {
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

        best_match.map(|(sim, entry)| (entry.response.clone(), entry.usage.clone(), sim))
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

/// Prefetch cache for speculative prefetching results
struct PrefetchCache {
    entries: HashMap<String, (Message, ProviderUsage, Instant)>,
    ttl: Duration,
}

impl PrefetchCache {
    fn new(ttl_seconds: u64) -> Self {
        Self {
            entries: HashMap::new(),
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    fn get(&mut self, key: &str) -> Option<(Message, ProviderUsage)> {
        if let Some((msg, usage, created)) = self.entries.remove(key) {
            if created.elapsed() < self.ttl {
                return Some((msg, usage));
            }
        }
        None
    }

    fn put(&mut self, key: String, response: Message, usage: ProviderUsage) {
        self.entries.insert(key, (response, usage, Instant::now()));
    }

    fn cleanup_expired(&mut self) {
        self.entries.retain(|_, (_, _, created)| created.elapsed() < self.ttl);
    }
}

/// Priority-based request scheduler
struct PriorityScheduler {
    hot_semaphore: Arc<Semaphore>,
    cold_semaphore: Arc<Semaphore>,
}

impl PriorityScheduler {
    fn new(max_hot: usize, max_cold: usize) -> Self {
        Self {
            hot_semaphore: Arc::new(Semaphore::new(max_hot)),
            cold_semaphore: Arc::new(Semaphore::new(max_cold)),
        }
    }

    /// Acquire a permit based on priority. Hot requests get more permits.
    async fn acquire(&self, priority: AgentPriority) -> PriorityPermit {
        match priority {
            AgentPriority::Hot => {
                let permit = self.hot_semaphore.clone().acquire_owned().await.unwrap();
                PriorityPermit::Hot(permit)
            }
            AgentPriority::Warm => {
                // Warm tries hot first, falls back to cold
                if let Ok(permit) = self.hot_semaphore.clone().try_acquire_owned() {
                    PriorityPermit::Hot(permit)
                } else {
                    let permit = self.cold_semaphore.clone().acquire_owned().await.unwrap();
                    PriorityPermit::Cold(permit)
                }
            }
            AgentPriority::Cold => {
                let permit = self.cold_semaphore.clone().acquire_owned().await.unwrap();
                PriorityPermit::Cold(permit)
            }
        }
    }
}

enum PriorityPermit {
    Hot(tokio::sync::OwnedSemaphorePermit),
    Cold(tokio::sync::OwnedSemaphorePermit),
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
    inner: Arc<P>,
    /// Configuration
    config: OptimizedProviderConfig,
    /// Semantic cache (shared across requests)
    cache: Arc<RwLock<SemanticCache>>,
    /// Prefetch cache for speculative results
    prefetch_cache: Arc<RwLock<PrefetchCache>>,
    /// Priority scheduler
    scheduler: Arc<PriorityScheduler>,
    /// Statistics
    stats: Arc<RwLock<OptimizationStats>>,
}

impl<P: Provider + Send + Sync + 'static> OptimizedProvider<P> {
    /// Create a new optimized provider wrapping the given provider
    pub fn new(inner: P, config: OptimizedProviderConfig) -> Self {
        let cache = SemanticCache::new(config.max_cache_entries, config.cache_ttl_seconds);
        let prefetch_cache = PrefetchCache::new(60); // 60 second TTL for prefetch
        let scheduler = PriorityScheduler::new(MAX_CONCURRENT_HOT, MAX_CONCURRENT_COLD);

        Self {
            inner: Arc::new(inner),
            config,
            cache: Arc::new(RwLock::new(cache)),
            prefetch_cache: Arc::new(RwLock::new(prefetch_cache)),
            scheduler: Arc::new(scheduler),
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

    /// Compute embedding for the query
    /// NOTE: This is a placeholder. In production, integrate with a real embedding model
    /// like sentence-transformers via Python bindings or an embedding API.
    fn compute_embedding(text: &str) -> Vec<f32> {
        // Placeholder: bag-of-words embedding
        // TODO: Replace with real embedding model integration
        let lowercase = text.to_lowercase();
        let words: Vec<&str> = lowercase.split_whitespace().collect();
        let mut embedding = vec![0.0f32; 384];

        for (i, word) in words.iter().enumerate() {
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

    /// Start a speculative prefetch in the background
    fn start_prefetch(
        &self,
        stage: WorkflowStage,
        context: String,
        model_config: ModelConfig,
        session_id: String,
        system: String,
    ) {
        if !self.config.enable_speculative_prefetch {
            return;
        }

        let Some(prefetch_query) = stage.generate_prefetch_query(&context) else {
            return;
        };

        let inner = Arc::clone(&self.inner);
        let prefetch_cache = Arc::clone(&self.prefetch_cache);
        let stats = Arc::clone(&self.stats);
        let predicted_stage = stage.predict_next().unwrap();

        tokio::spawn(async move {
            // Create a simple prefetch message
            let prefetch_msg = Message::new(
                Role::User,
                Utc::now().timestamp(),
                vec![MessageContent::Text(TextContent {
                    raw: RawTextContent {
                        text: prefetch_query.clone(),
                        meta: None,
                    },
                    annotations: None,
                })],
            );

            // Execute prefetch request
            match inner.complete(&model_config, &session_id, &system, &[prefetch_msg], &[]).await {
                Ok((response, usage)) => {
                    let cache_key = format!("{:?}", predicted_stage);
                    prefetch_cache.write().await.put(cache_key, response, usage);
                    tracing::debug!("Prefetch completed for {:?}", predicted_stage);
                }
                Err(e) => {
                    tracing::debug!("Prefetch failed: {:?}", e);
                }
            }

            // Update stats
            let mut s = stats.write().await;
            s.prefetch_misses += 1; // Will be corrected to hit if used
        });
    }
}

#[async_trait]
impl<P: Provider + Send + Sync + 'static> Provider for OptimizedProvider<P> {
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
        // But we still apply priority scheduling
        let stage = WorkflowStage::classify(messages);
        let priority = stage.priority();

        if self.config.enable_priority_scheduling {
            let _permit = self.scheduler.acquire(priority).await;
            self.inner.stream(model_config, session_id, system, messages, tools).await
        } else {
            self.inner.stream(model_config, session_id, system, messages, tools).await
        }
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

        // Check prefetch cache first (speculative prefetching)
        if self.config.enable_speculative_prefetch {
            let cache_key = format!("{:?}", stage);
            if let Some((response, usage)) = self.prefetch_cache.write().await.get(&cache_key) {
                let mut stats = self.stats.write().await;
                stats.prefetch_hits += 1;
                stats.prefetch_misses = stats.prefetch_misses.saturating_sub(1);
                tracing::info!("Prefetch hit for {:?}", stage);
                return Ok((response, usage));
            }
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

            if let Some((response, usage, similarity)) = cached {
                // Cache hit!
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;
                tracing::info!(
                    "Semantic cache hit for session {} (stage: {:?}, similarity: {:.3})",
                    session_id,
                    stage,
                    similarity
                );
                return Ok((response, usage));
            }

            // Acquire priority permit before calling LLM
            let _permit = if self.config.enable_priority_scheduling {
                Some(self.scheduler.acquire(priority).await)
            } else {
                None
            };

            // Cache miss - call the underlying provider
            let result = self.inner.complete(model_config, session_id, system, messages, tools).await?;

            // Start speculative prefetch for predicted next stage
            self.start_prefetch(
                stage,
                query_text.clone(),
                model_config.clone(),
                session_id.to_string(),
                system.to_string(),
            );

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
            // Caching disabled, just apply priority scheduling and forward
            let _permit = if self.config.enable_priority_scheduling {
                Some(self.scheduler.acquire(priority).await)
            } else {
                None
            };

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
        assert_eq!(WorkflowStage::TestAnalyze.priority(), AgentPriority::Hot);
        assert_eq!(WorkflowStage::CodeEdit.priority(), AgentPriority::Hot);
        assert_eq!(WorkflowStage::Explore.priority(), AgentPriority::Cold);
        assert_eq!(WorkflowStage::Docs.priority(), AgentPriority::Cold);
    }

    #[test]
    fn test_workflow_prediction() {
        assert_eq!(WorkflowStage::CodeEdit.predict_next(), Some(WorkflowStage::TestAnalyze));
        assert_eq!(WorkflowStage::TestAnalyze.predict_next(), Some(WorkflowStage::CodeEdit));
        assert_eq!(WorkflowStage::Plan.predict_next(), Some(WorkflowStage::CodeEdit));
        assert_eq!(WorkflowStage::Explore.predict_next(), None);
    }

    #[test]
    fn test_prefetch_query_generation() {
        let context = "def add(a, b): return a + b";
        assert!(WorkflowStage::CodeEdit.generate_prefetch_query(context).is_some());
        assert!(WorkflowStage::TestAnalyze.generate_prefetch_query(context).is_some());
        assert!(WorkflowStage::Explore.generate_prefetch_query(context).is_none());
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
    fn test_semantic_cache_operations() {
        use super::super::base::Usage;

        let mut cache = SemanticCache::new(10, 300);

        // Create a mock message and usage
        let embedding = vec![1.0, 0.0, 0.0];
        let response = Message::new(Role::Assistant, 0, vec![]);
        let usage = ProviderUsage::new("test".to_string(), Usage::default());

        // Put and get
        cache.put(embedding.clone(), "test query".to_string(), response.clone(), usage.clone());
        assert_eq!(cache.len(), 1);

        // Should hit with same embedding
        let result = cache.get(&embedding, 0.85);
        assert!(result.is_some());

        // Should miss with orthogonal embedding
        let orthogonal = vec![0.0, 1.0, 0.0];
        let result = cache.get(&orthogonal, 0.85);
        assert!(result.is_none());
    }

    #[test]
    fn test_prefetch_cache_operations() {
        use super::super::base::Usage;

        let mut cache = PrefetchCache::new(60);

        let response = Message::new(Role::Assistant, 0, vec![]);
        let usage = ProviderUsage::new("test".to_string(), Usage::default());

        cache.put("TestAnalyze".to_string(), response, usage);

        // Should hit
        assert!(cache.get("TestAnalyze").is_some());

        // Should miss (already consumed)
        assert!(cache.get("TestAnalyze").is_none());
    }

    #[tokio::test]
    async fn test_priority_scheduler() {
        let scheduler = PriorityScheduler::new(2, 1);

        // Hot should get permit immediately
        let _permit1 = scheduler.acquire(AgentPriority::Hot).await;
        let _permit2 = scheduler.acquire(AgentPriority::Hot).await;

        // Third hot should still work (semaphore)
        // Cold should work with cold semaphore
        let _permit3 = scheduler.acquire(AgentPriority::Cold).await;
    }

    #[test]
    fn test_compute_embedding() {
        let emb1 = OptimizedProvider::<DummyProvider>::compute_embedding("hello world");
        let emb2 = OptimizedProvider::<DummyProvider>::compute_embedding("hello world");
        let emb3 = OptimizedProvider::<DummyProvider>::compute_embedding("goodbye moon");

        // Same text should produce same embedding
        assert_eq!(emb1, emb2);

        // Different text should produce different embedding
        assert_ne!(emb1, emb3);

        // Should be normalized (length ~= 1)
        let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    // Dummy provider for testing compute_embedding
    struct DummyProvider;

    #[async_trait]
    impl Provider for DummyProvider {
        fn get_name(&self) -> &str { "dummy" }
        fn get_model_config(&self) -> ModelConfig {
            ModelConfig::new("test").expect("test config")
        }
        async fn stream(&self, _: &ModelConfig, _: &str, _: &str, _: &[Message], _: &[Tool])
            -> Result<MessageStream, ProviderError> { unimplemented!() }
        async fn complete(&self, _: &ModelConfig, _: &str, _: &str, _: &[Message], _: &[Tool])
            -> Result<(Message, ProviderUsage), ProviderError> { unimplemented!() }
        async fn supports_cache_control(&self) -> bool { false }
        fn supports_embeddings(&self) -> bool { false }
        async fn create_embeddings(&self, _: &str, _: Vec<String>)
            -> Result<Vec<Vec<f32>>, ProviderError> { unimplemented!() }
    }
}
