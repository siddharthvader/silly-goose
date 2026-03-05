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
//! Embedding Strategy:
//! 1. Try the inference server's embedding endpoint (vLLM, Ollama support this)
//! 2. Fall back to TF-IDF/BM25 (works well for code queries)
//! 3. Optionally use OpenAI embeddings if configured

use super::base::{MessageStream, Provider, ProviderUsage};
use super::errors::ProviderError;
use crate::conversation::message::{Message, MessageContent};
use crate::model::ModelConfig;
use async_trait::async_trait;
use chrono::Utc;
use rmcp::model::{RawTextContent, Role, Tool, TextContent};
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

/// Embedding dimension for TF-IDF fallback
const TFIDF_EMBEDDING_DIM: usize = 384;

/// Embedding provider configuration
#[derive(Debug, Clone)]
pub enum EmbeddingSource {
    /// Use the same inference server (vLLM, Ollama) - recommended
    InferenceServer,
    /// Use OpenAI embeddings API
    OpenAI { api_key: String, model: String },
    /// Use TF-IDF fallback only (no external calls)
    TfIdfOnly,
    /// Auto-detect: try inference server, fall back to TF-IDF
    Auto,
}

impl Default for EmbeddingSource {
    fn default() -> Self {
        Self::Auto
    }
}

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
            .find(|m| m.role == Role::User)
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
    pub embedding_source_used: String,
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

    #[allow(dead_code)]
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

#[allow(dead_code)]
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
    /// Embedding source configuration
    pub embedding_source: EmbeddingSource,
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
            embedding_source: EmbeddingSource::Auto,
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
    /// Whether the inner provider supports embeddings (cached)
    inner_supports_embeddings: bool,
}

impl<P: Provider + Send + Sync + 'static> OptimizedProvider<P> {
    /// Create a new optimized provider wrapping the given provider
    pub fn new(inner: P, config: OptimizedProviderConfig) -> Self {
        let inner_supports_embeddings = inner.supports_embeddings();
        let cache = SemanticCache::new(config.max_cache_entries, config.cache_ttl_seconds);
        let prefetch_cache = PrefetchCache::new(60);
        let scheduler = PriorityScheduler::new(MAX_CONCURRENT_HOT, MAX_CONCURRENT_COLD);

        Self {
            inner: Arc::new(inner),
            config,
            cache: Arc::new(RwLock::new(cache)),
            prefetch_cache: Arc::new(RwLock::new(prefetch_cache)),
            scheduler: Arc::new(scheduler),
            stats: Arc::new(RwLock::new(OptimizationStats::default())),
            inner_supports_embeddings,
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
            .find(|m| m.role == Role::User)
            .map(|m| m.as_concat_text())
            .unwrap_or_default()
    }

    /// Compute embedding using the configured source
    async fn compute_embedding(&self, text: &str, session_id: &str) -> Vec<f32> {
        match &self.config.embedding_source {
            EmbeddingSource::InferenceServer => {
                self.compute_embedding_from_server(text, session_id).await
            }
            EmbeddingSource::OpenAI { api_key, model } => {
                self.compute_embedding_from_openai(text, api_key, model).await
            }
            EmbeddingSource::TfIdfOnly => {
                Self::compute_tfidf_embedding(text)
            }
            EmbeddingSource::Auto => {
                // Try inference server first, fall back to TF-IDF
                if self.inner_supports_embeddings {
                    self.compute_embedding_from_server(text, session_id).await
                } else {
                    Self::compute_tfidf_embedding(text)
                }
            }
        }
    }

    /// Compute embedding from the inference server (vLLM, Ollama, etc.)
    async fn compute_embedding_from_server(&self, text: &str, session_id: &str) -> Vec<f32> {
        match self.inner.create_embeddings(session_id, vec![text.to_string()]).await {
            Ok(embeddings) if !embeddings.is_empty() => {
                tracing::debug!("Got embedding from inference server");
                embeddings.into_iter().next().unwrap_or_else(|| Self::compute_tfidf_embedding(text))
            }
            Ok(_) => {
                tracing::debug!("Empty embedding from server, using TF-IDF fallback");
                Self::compute_tfidf_embedding(text)
            }
            Err(e) => {
                tracing::debug!("Embedding from server failed: {:?}, using TF-IDF fallback", e);
                Self::compute_tfidf_embedding(text)
            }
        }
    }

    /// Compute embedding from OpenAI API
    async fn compute_embedding_from_openai(&self, text: &str, api_key: &str, model: &str) -> Vec<f32> {
        // Use reqwest to call OpenAI embeddings API
        let client = reqwest::Client::new();

        #[derive(serde::Serialize)]
        struct EmbeddingRequest {
            input: String,
            model: String,
        }

        #[derive(serde::Deserialize)]
        struct EmbeddingResponse {
            data: Vec<EmbeddingData>,
        }

        #[derive(serde::Deserialize)]
        struct EmbeddingData {
            embedding: Vec<f32>,
        }

        let request = EmbeddingRequest {
            input: text.to_string(),
            model: model.to_string(),
        };

        match client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&request)
            .send()
            .await
        {
            Ok(response) => {
                match response.json::<EmbeddingResponse>().await {
                    Ok(data) if !data.data.is_empty() => {
                        tracing::debug!("Got embedding from OpenAI");
                        data.data.into_iter().next().unwrap().embedding
                    }
                    _ => {
                        tracing::debug!("OpenAI embedding failed, using TF-IDF fallback");
                        Self::compute_tfidf_embedding(text)
                    }
                }
            }
            Err(e) => {
                tracing::debug!("OpenAI request failed: {:?}, using TF-IDF fallback", e);
                Self::compute_tfidf_embedding(text)
            }
        }
    }

    /// Compute TF-IDF based embedding (fallback, works well for code queries)
    ///
    /// This is surprisingly effective for code-related queries because:
    /// - Code queries often share key identifiers (function names, class names)
    /// - "What does authenticate do?" and "Explain authenticate function" both contain "authenticate"
    /// - The hash-based approach captures word importance without external dependencies
    fn compute_tfidf_embedding(text: &str) -> Vec<f32> {
        let lowercase = text.to_lowercase();

        // Tokenize with better handling of code identifiers
        let words: Vec<&str> = lowercase
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| !w.is_empty() && w.len() > 1)
            .collect();

        let mut embedding = vec![0.0f32; TFIDF_EMBEDDING_DIM];
        let total_words = words.len() as f32;

        // Count word frequencies
        let mut word_counts: HashMap<&str, usize> = HashMap::new();
        for word in &words {
            *word_counts.entry(word).or_insert(0) += 1;
        }

        // Build embedding with TF-IDF-like weighting
        for (word, count) in word_counts {
            // Term frequency (log-scaled)
            let tf = (1.0 + (count as f32).ln()) / total_words.max(1.0);

            // IDF approximation using hash (common words hash to same buckets, reducing their impact)
            let hash = word.bytes().fold(0u64, |acc, b| {
                acc.wrapping_mul(31).wrapping_add(b as u64)
            });

            // Use multiple hash positions for better distribution
            let idx1 = (hash % TFIDF_EMBEDDING_DIM as u64) as usize;
            let idx2 = ((hash >> 8) % TFIDF_EMBEDDING_DIM as u64) as usize;
            let idx3 = ((hash >> 16) % TFIDF_EMBEDDING_DIM as u64) as usize;

            // Weight by word length (longer words are often more specific)
            let length_weight = (word.len() as f32).sqrt() / 3.0;

            embedding[idx1] += tf * length_weight;
            embedding[idx2] += tf * length_weight * 0.5;
            embedding[idx3] += tf * length_weight * 0.25;
        }

        // L2 normalize
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
            s.prefetch_misses += 1;
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
            let query_embedding = self.compute_embedding(&query_text, session_id).await;

            // Update embedding source stat
            {
                let mut stats = self.stats.write().await;
                stats.embedding_source_used = match &self.config.embedding_source {
                    EmbeddingSource::InferenceServer => "inference_server".to_string(),
                    EmbeddingSource::OpenAI { .. } => "openai".to_string(),
                    EmbeddingSource::TfIdfOnly => "tfidf".to_string(),
                    EmbeddingSource::Auto => {
                        if self.inner_supports_embeddings {
                            "auto:inference_server".to_string()
                        } else {
                            "auto:tfidf".to_string()
                        }
                    }
                };
            }

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
        // We always "support" embeddings via our fallback
        true
    }

    async fn create_embeddings(
        &self,
        session_id: &str,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            let emb = self.compute_embedding(&text, session_id).await;
            embeddings.push(emb);
        }
        Ok(embeddings)
    }
}

/// Check if a provider should use optimizations (i.e., is self-hosted)
pub fn should_optimize(provider_name: &str) -> bool {
    let name = provider_name.to_lowercase();

    // Direct match for known self-hosted providers
    if matches!(
        name.as_str(),
        "ollama" | "openai-compatible" | "vllm" | "local" | "lm-studio"
    ) {
        return true;
    }

    // Check if "openai" provider is pointed at a custom host (not api.openai.com)
    if name == "openai" {
        if let Ok(host) = std::env::var("OPENAI_HOST") {
            let host_lower = host.to_lowercase();
            // If OPENAI_HOST is set to something other than OpenAI's API, treat as self-hosted
            if !host_lower.contains("api.openai.com") && !host_lower.is_empty() {
                tracing::info!("OpenAI provider with custom host {} - enabling optimizations", host);
                return true;
            }
        }
    }

    false
}

// ============================================================================
// Dynamic Provider Wrapper (for wrapping Arc<dyn Provider>)
// ============================================================================

/// Optimized provider that wraps Arc<dyn Provider> directly
/// This is used by init.rs to wrap providers without knowing the concrete type
pub struct OptimizedProviderDyn {
    /// The underlying provider to wrap
    inner: Arc<dyn Provider>,
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
    /// Whether the inner provider supports embeddings (cached)
    inner_supports_embeddings: bool,
}

impl OptimizedProviderDyn {
    /// Create a new optimized provider wrapping the given Arc<dyn Provider>
    pub fn new(inner: Arc<dyn Provider>, config: OptimizedProviderConfig) -> Self {
        let inner_supports_embeddings = inner.supports_embeddings();
        let cache = SemanticCache::new(config.max_cache_entries, config.cache_ttl_seconds);
        let prefetch_cache = PrefetchCache::new(60);
        let scheduler = PriorityScheduler::new(MAX_CONCURRENT_HOT, MAX_CONCURRENT_COLD);

        tracing::info!(
            "Creating OptimizedProviderDyn wrapping {} (semantic_cache={}, priority={}, prefetch={}, embedding_source={:?})",
            inner.get_name(),
            config.enable_semantic_cache,
            config.enable_priority_scheduling,
            config.enable_speculative_prefetch,
            config.embedding_source
        );

        Self {
            inner,
            config,
            cache: Arc::new(RwLock::new(cache)),
            prefetch_cache: Arc::new(RwLock::new(prefetch_cache)),
            scheduler: Arc::new(scheduler),
            stats: Arc::new(RwLock::new(OptimizationStats::default())),
            inner_supports_embeddings,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(inner: Arc<dyn Provider>) -> Self {
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
            .find(|m| m.role == Role::User)
            .map(|m| m.as_concat_text())
            .unwrap_or_default()
    }

    /// Compute embedding using the configured source
    async fn compute_embedding(&self, text: &str, session_id: &str) -> Vec<f32> {
        match &self.config.embedding_source {
            EmbeddingSource::InferenceServer => {
                self.compute_embedding_from_server(text, session_id).await
            }
            EmbeddingSource::OpenAI { api_key, model } => {
                self.compute_embedding_from_openai(text, api_key, model).await
            }
            EmbeddingSource::TfIdfOnly => {
                Self::compute_tfidf_embedding(text)
            }
            EmbeddingSource::Auto => {
                if self.inner_supports_embeddings {
                    self.compute_embedding_from_server(text, session_id).await
                } else {
                    Self::compute_tfidf_embedding(text)
                }
            }
        }
    }

    /// Compute embedding from the inference server
    async fn compute_embedding_from_server(&self, text: &str, session_id: &str) -> Vec<f32> {
        match self.inner.create_embeddings(session_id, vec![text.to_string()]).await {
            Ok(embeddings) if !embeddings.is_empty() => {
                tracing::debug!("Got embedding from inference server");
                embeddings.into_iter().next().unwrap_or_else(|| Self::compute_tfidf_embedding(text))
            }
            Ok(_) => {
                tracing::debug!("Empty embedding from server, using TF-IDF fallback");
                Self::compute_tfidf_embedding(text)
            }
            Err(e) => {
                tracing::debug!("Embedding from server failed: {:?}, using TF-IDF fallback", e);
                Self::compute_tfidf_embedding(text)
            }
        }
    }

    /// Compute embedding from OpenAI API
    async fn compute_embedding_from_openai(&self, text: &str, api_key: &str, model: &str) -> Vec<f32> {
        let client = reqwest::Client::new();

        #[derive(serde::Serialize)]
        struct EmbeddingRequest {
            input: String,
            model: String,
        }

        #[derive(serde::Deserialize)]
        struct EmbeddingResponse {
            data: Vec<EmbeddingData>,
        }

        #[derive(serde::Deserialize)]
        struct EmbeddingData {
            embedding: Vec<f32>,
        }

        let request = EmbeddingRequest {
            input: text.to_string(),
            model: model.to_string(),
        };

        match client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&request)
            .send()
            .await
        {
            Ok(response) => {
                match response.json::<EmbeddingResponse>().await {
                    Ok(data) if !data.data.is_empty() => {
                        tracing::debug!("Got embedding from OpenAI");
                        data.data.into_iter().next().unwrap().embedding
                    }
                    _ => {
                        tracing::debug!("OpenAI embedding failed, using TF-IDF fallback");
                        Self::compute_tfidf_embedding(text)
                    }
                }
            }
            Err(e) => {
                tracing::debug!("OpenAI request failed: {:?}, using TF-IDF fallback", e);
                Self::compute_tfidf_embedding(text)
            }
        }
    }

    /// Compute TF-IDF based embedding (same as generic version)
    fn compute_tfidf_embedding(text: &str) -> Vec<f32> {
        let lowercase = text.to_lowercase();
        let words: Vec<&str> = lowercase
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| !w.is_empty() && w.len() > 1)
            .collect();

        let mut embedding = vec![0.0f32; TFIDF_EMBEDDING_DIM];
        let total_words = words.len() as f32;

        let mut word_counts: HashMap<&str, usize> = HashMap::new();
        for word in &words {
            *word_counts.entry(word).or_insert(0) += 1;
        }

        for (word, count) in word_counts {
            let tf = (1.0 + (count as f32).ln()) / total_words.max(1.0);
            let hash = word.bytes().fold(0u64, |acc, b| {
                acc.wrapping_mul(31).wrapping_add(b as u64)
            });

            let idx1 = (hash % TFIDF_EMBEDDING_DIM as u64) as usize;
            let idx2 = ((hash >> 8) % TFIDF_EMBEDDING_DIM as u64) as usize;
            let idx3 = ((hash >> 16) % TFIDF_EMBEDDING_DIM as u64) as usize;

            let length_weight = (word.len() as f32).sqrt() / 3.0;

            embedding[idx1] += tf * length_weight;
            embedding[idx2] += tf * length_weight * 0.5;
            embedding[idx3] += tf * length_weight * 0.25;
        }

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

            let mut s = stats.write().await;
            s.prefetch_misses += 1;
        });
    }
}

#[async_trait]
impl Provider for OptimizedProviderDyn {
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

        let stage = WorkflowStage::classify(messages);
        let priority = stage.priority();

        // Update priority stats
        {
            let mut stats = self.stats.write().await;
            let key = format!("{:?}", priority);
            *stats.requests_by_priority.entry(key).or_insert(0) += 1;
        }

        // Check prefetch cache first
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

        // Check semantic cache
        if self.config.enable_semantic_cache {
            let query_text = Self::extract_query_text(messages);
            let query_embedding = self.compute_embedding(&query_text, session_id).await;

            // Update embedding source stat
            {
                let mut stats = self.stats.write().await;
                stats.embedding_source_used = match &self.config.embedding_source {
                    EmbeddingSource::InferenceServer => "inference_server".to_string(),
                    EmbeddingSource::OpenAI { .. } => "openai".to_string(),
                    EmbeddingSource::TfIdfOnly => "tfidf".to_string(),
                    EmbeddingSource::Auto => {
                        if self.inner_supports_embeddings {
                            "auto:inference_server".to_string()
                        } else {
                            "auto:tfidf".to_string()
                        }
                    }
                };
            }

            // Try cache lookup
            let cached = {
                let mut cache = self.cache.write().await;
                cache.get(&query_embedding, self.config.similarity_threshold)
            };

            if let Some((response, usage, similarity)) = cached {
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

            // Start speculative prefetch
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
        true
    }

    async fn create_embeddings(
        &self,
        session_id: &str,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            let emb = self.compute_embedding(&text, session_id).await;
            embeddings.push(emb);
        }
        Ok(embeddings)
    }
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
    fn test_tfidf_embedding() {
        let emb1 = OptimizedProvider::<DummyProvider>::compute_tfidf_embedding("What does authenticate do?");
        let emb2 = OptimizedProvider::<DummyProvider>::compute_tfidf_embedding("Explain the authenticate function");
        let emb3 = OptimizedProvider::<DummyProvider>::compute_tfidf_embedding("How do I install Docker?");

        // Similar queries should have high similarity
        let sim_similar = SemanticCache::cosine_similarity(&emb1, &emb2);
        // Different queries should have low similarity
        let sim_different = SemanticCache::cosine_similarity(&emb1, &emb3);

        println!("Similar query similarity: {}", sim_similar);
        println!("Different query similarity: {}", sim_different);

        // Similar queries share "authenticate" - should be reasonably similar
        assert!(sim_similar > sim_different, "Similar queries should have higher similarity");

        // Embedding should be normalized
        let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");
    }

    #[test]
    fn test_tfidf_embedding_code_queries() {
        // Test that code-specific queries work well
        let emb1 = OptimizedProvider::<DummyProvider>::compute_tfidf_embedding(
            "What does the UserAuthService.authenticate method do?"
        );
        let emb2 = OptimizedProvider::<DummyProvider>::compute_tfidf_embedding(
            "Explain UserAuthService authenticate"
        );
        let emb3 = OptimizedProvider::<DummyProvider>::compute_tfidf_embedding(
            "How does the payment processing work?"
        );

        let sim_same_topic = SemanticCache::cosine_similarity(&emb1, &emb2);
        let sim_different_topic = SemanticCache::cosine_similarity(&emb1, &emb3);

        assert!(sim_same_topic > sim_different_topic);
    }

    #[test]
    fn test_semantic_cache_operations() {
        use super::super::base::Usage;

        let mut cache = SemanticCache::new(10, 300);

        let embedding = vec![1.0, 0.0, 0.0];
        let response = Message::new(Role::Assistant, 0, vec![]);
        let usage = ProviderUsage::new("test".to_string(), Usage::default());

        cache.put(embedding.clone(), "test query".to_string(), response.clone(), usage.clone());
        assert_eq!(cache.len(), 1);

        let result = cache.get(&embedding, 0.85);
        assert!(result.is_some());

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

        assert!(cache.get("TestAnalyze").is_some());
        assert!(cache.get("TestAnalyze").is_none());
    }

    #[tokio::test]
    async fn test_priority_scheduler() {
        let scheduler = PriorityScheduler::new(2, 1);

        let _permit1 = scheduler.acquire(AgentPriority::Hot).await;
        let _permit2 = scheduler.acquire(AgentPriority::Hot).await;
        let _permit3 = scheduler.acquire(AgentPriority::Cold).await;
    }

    // Dummy provider for testing
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
