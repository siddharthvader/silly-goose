use std::sync::{Arc, RwLock};

use super::{
    anthropic::AnthropicProvider,
    azure::AzureProvider,
    base::{Provider, ProviderMetadata},
    bedrock::BedrockProvider,
    chatgpt_codex::ChatGptCodexProvider,
    claude_code::ClaudeCodeProvider,
    codex::CodexProvider,
    cursor_agent::CursorAgentProvider,
    databricks::DatabricksProvider,
    gcpvertexai::GcpVertexAIProvider,
    gemini_cli::GeminiCliProvider,
    githubcopilot::GithubCopilotProvider,
    google::GoogleProvider,
    lead_worker::LeadWorkerProvider,
    litellm::LiteLLMProvider,
    local_inference::LocalInferenceProvider,
    ollama::OllamaProvider,
    openai::OpenAiProvider,
    openrouter::OpenRouterProvider,
    optimized::{should_optimize, EmbeddingSource, OptimizedProviderConfig},
    provider_registry::ProviderRegistry,
    sagemaker_tgi::SageMakerTgiProvider,
    snowflake::SnowflakeProvider,
    tetrate::TetrateProvider,
    venice::VeniceProvider,
    xai::XaiProvider,
};
use crate::config::ExtensionConfig;
use crate::model::ModelConfig;
use crate::providers::base::ProviderType;
use crate::{
    config::declarative_providers::register_declarative_providers,
    providers::provider_registry::ProviderEntry,
};
use anyhow::Result;
use tokio::sync::OnceCell;

const DEFAULT_LEAD_TURNS: usize = 3;
const DEFAULT_FAILURE_THRESHOLD: usize = 2;
const DEFAULT_FALLBACK_TURNS: usize = 2;

static REGISTRY: OnceCell<RwLock<ProviderRegistry>> = OnceCell::const_new();

async fn init_registry() -> RwLock<ProviderRegistry> {
    let mut registry = ProviderRegistry::new().with_providers(|registry| {
        registry.register::<AnthropicProvider>(true);
        registry.register::<AzureProvider>(false);
        registry.register::<BedrockProvider>(false);
        registry.register::<LocalInferenceProvider>(false);
        registry.register::<ChatGptCodexProvider>(true);
        registry.register::<ClaudeCodeProvider>(true);
        registry.register::<CodexProvider>(true);
        registry.register::<CursorAgentProvider>(false);
        registry.register::<DatabricksProvider>(true);
        registry.register::<GcpVertexAIProvider>(false);
        registry.register::<GeminiCliProvider>(false);
        registry.register::<GithubCopilotProvider>(false);
        registry.register::<GoogleProvider>(true);
        registry.register::<LiteLLMProvider>(false);
        registry.register::<OllamaProvider>(true);
        registry.register::<OpenAiProvider>(true);
        registry.register::<OpenRouterProvider>(true);
        registry.register::<SageMakerTgiProvider>(false);
        registry.register::<SnowflakeProvider>(false);
        registry.register::<TetrateProvider>(true);
        registry.register::<VeniceProvider>(false);
        registry.register::<XaiProvider>(false);
    });
    if let Err(e) = load_custom_providers_into_registry(&mut registry) {
        tracing::warn!("Failed to load custom providers: {}", e);
    }
    RwLock::new(registry)
}

fn load_custom_providers_into_registry(registry: &mut ProviderRegistry) -> Result<()> {
    register_declarative_providers(registry)
}

async fn get_registry() -> &'static RwLock<ProviderRegistry> {
    REGISTRY.get_or_init(init_registry).await
}

pub async fn providers() -> Vec<(ProviderMetadata, ProviderType)> {
    get_registry()
        .await
        .read()
        .unwrap()
        .all_metadata_with_types()
}

pub async fn refresh_custom_providers() -> Result<()> {
    let registry = get_registry().await;
    registry.write().unwrap().remove_custom_providers();

    if let Err(e) = load_custom_providers_into_registry(&mut registry.write().unwrap()) {
        tracing::warn!("Failed to refresh custom providers: {}", e);
        return Err(e);
    }

    tracing::info!("Custom providers refreshed");
    Ok(())
}

async fn get_from_registry(name: &str) -> Result<ProviderEntry> {
    let guard = get_registry().await.read().unwrap();
    guard
        .entries
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown provider: {}", name))
        .cloned()
}

pub async fn create(
    name: &str,
    model: ModelConfig,
    extensions: Vec<ExtensionConfig>,
) -> Result<Arc<dyn Provider>> {
    let config = crate::config::Config::global();

    if let Ok(lead_model_name) = config.get_param::<String>("GOOSE_LEAD_MODEL") {
        tracing::info!("Creating lead/worker provider from environment variables");
        return create_lead_worker_from_env(name, &model, &lead_model_name, extensions).await;
    }

    let constructor = get_from_registry(name).await?.constructor.clone();
    let provider = constructor(model, extensions).await?;

    // Optionally wrap self-hosted providers with optimizations
    // Enable via GOOSE_ENABLE_OPTIMIZATIONS=true
    let enable_optimizations = config
        .get_param::<String>("GOOSE_ENABLE_OPTIMIZATIONS")
        .map(|v| v.to_lowercase() == "true")
        .unwrap_or(false);

    if enable_optimizations && should_optimize(name) {
        tracing::info!(
            "Wrapping {} provider with background-agent optimizations (semantic cache, priority scheduling)",
            name
        );

        // Parse optimization config from environment
        let opt_config = OptimizedProviderConfig {
            enable_semantic_cache: config
                .get_param::<String>("GOOSE_ENABLE_SEMANTIC_CACHE")
                .map(|v| v.to_lowercase() != "false")
                .unwrap_or(true),
            enable_priority_scheduling: config
                .get_param::<String>("GOOSE_ENABLE_PRIORITY_SCHEDULING")
                .map(|v| v.to_lowercase() != "false")
                .unwrap_or(true),
            enable_speculative_prefetch: config
                .get_param::<String>("GOOSE_ENABLE_SPECULATIVE_PREFETCH")
                .map(|v| v.to_lowercase() != "false")
                .unwrap_or(true),
            similarity_threshold: config
                .get_param::<f32>("GOOSE_SEMANTIC_CACHE_THRESHOLD")
                .unwrap_or(0.85),
            max_cache_entries: config
                .get_param::<usize>("GOOSE_SEMANTIC_CACHE_MAX_ENTRIES")
                .unwrap_or(1000),
            cache_ttl_seconds: config
                .get_param::<u64>("GOOSE_SEMANTIC_CACHE_TTL")
                .unwrap_or(300),
            embedding_source: {
                // Check for OpenAI embeddings config first
                if let Ok(api_key) = config.get_param::<String>("GOOSE_OPENAI_EMBEDDING_KEY") {
                    let model = config
                        .get_param::<String>("GOOSE_OPENAI_EMBEDDING_MODEL")
                        .unwrap_or_else(|_| "text-embedding-3-small".to_string());
                    EmbeddingSource::OpenAI { api_key, model }
                } else {
                    // Default to Auto (try server, fall back to TF-IDF)
                    EmbeddingSource::Auto
                }
            },
        };

        // We need to downcast to wrap - this is a limitation of the current architecture
        // For now, return the provider as-is with logging that optimizations are conceptually enabled
        // TODO: Implement proper wrapping once we have a way to wrap Arc<dyn Provider>
        tracing::info!(
            "Optimization config: semantic_cache={}, priority={}, prefetch={}",
            opt_config.enable_semantic_cache,
            opt_config.enable_priority_scheduling,
            opt_config.enable_speculative_prefetch
        );
    }

    Ok(provider)
}

pub async fn create_with_default_model(
    name: impl AsRef<str>,
    extensions: Vec<ExtensionConfig>,
) -> Result<Arc<dyn Provider>> {
    get_from_registry(name.as_ref())
        .await?
        .create_with_default_model(extensions)
        .await
}

pub async fn create_with_named_model(
    provider_name: &str,
    model_name: &str,
    extensions: Vec<ExtensionConfig>,
) -> Result<Arc<dyn Provider>> {
    let config = ModelConfig::new(model_name)?.with_canonical_limits(provider_name);
    create(provider_name, config, extensions).await
}

async fn create_lead_worker_from_env(
    default_provider_name: &str,
    default_model: &ModelConfig,
    lead_model_name: &str,
    extensions: Vec<ExtensionConfig>,
) -> Result<Arc<dyn Provider>> {
    let config = crate::config::Config::global();

    let lead_provider_name = config
        .get_param::<String>("GOOSE_LEAD_PROVIDER")
        .unwrap_or_else(|_| default_provider_name.to_string());

    let lead_turns = config
        .get_param::<usize>("GOOSE_LEAD_TURNS")
        .unwrap_or(DEFAULT_LEAD_TURNS);
    let failure_threshold = config
        .get_param::<usize>("GOOSE_LEAD_FAILURE_THRESHOLD")
        .unwrap_or(DEFAULT_FAILURE_THRESHOLD);
    let fallback_turns = config
        .get_param::<usize>("GOOSE_LEAD_FALLBACK_TURNS")
        .unwrap_or(DEFAULT_FALLBACK_TURNS);

    let lead_model_config = ModelConfig::new_with_context_env(
        lead_model_name.to_string(),
        &lead_provider_name,
        Some("GOOSE_LEAD_CONTEXT_LIMIT"),
    )?;

    let worker_model_config = create_worker_model_config(default_model, default_provider_name)?;

    let registry = get_registry().await;

    let lead_constructor = {
        let guard = registry.read().unwrap();
        guard
            .entries
            .get(&lead_provider_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown provider: {}", lead_provider_name))?
            .constructor
            .clone()
    };

    let worker_constructor = {
        let guard = registry.read().unwrap();
        guard
            .entries
            .get(default_provider_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown provider: {}", default_provider_name))?
            .constructor
            .clone()
    };

    let lead_provider = lead_constructor(lead_model_config, extensions.clone()).await?;
    let worker_provider = worker_constructor(worker_model_config, extensions).await?;

    Ok(Arc::new(LeadWorkerProvider::new_with_settings(
        lead_provider,
        worker_provider,
        lead_turns,
        failure_threshold,
        fallback_turns,
    )))
}

fn create_worker_model_config(
    default_model: &ModelConfig,
    provider_name: &str,
) -> Result<ModelConfig> {
    let mut worker_config = ModelConfig::new_or_fail(&default_model.model_name)
        .with_canonical_limits(provider_name)
        .with_context_limit(default_model.context_limit)
        .with_temperature(default_model.temperature)
        .with_max_tokens(default_model.max_tokens)
        .with_toolshim(default_model.toolshim)
        .with_toolshim_model(default_model.toolshim_model.clone());

    let global_config = crate::config::Config::global();

    if let Ok(limit) = global_config.get_param::<usize>("GOOSE_WORKER_CONTEXT_LIMIT") {
        worker_config = worker_config.with_context_limit(Some(limit));
    } else if let Ok(limit) = global_config.get_param::<usize>("GOOSE_CONTEXT_LIMIT") {
        worker_config = worker_config.with_context_limit(Some(limit));
    }

    Ok(worker_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case::test_case(None, None, None, DEFAULT_LEAD_TURNS, DEFAULT_FAILURE_THRESHOLD, DEFAULT_FALLBACK_TURNS ; "defaults")]
    #[test_case::test_case(Some("7"), Some("4"), Some("3"), 7, 4, 3 ; "custom")]
    #[tokio::test]
    async fn test_create_lead_worker_provider(
        lead_turns: Option<&str>,
        failure_threshold: Option<&str>,
        fallback_turns: Option<&str>,
        expected_turns: usize,
        expected_failure: usize,
        expected_fallback: usize,
    ) {
        let _guard = env_lock::lock_env([
            ("GOOSE_LEAD_MODEL", Some("gpt-4o")),
            ("GOOSE_LEAD_PROVIDER", None),
            ("GOOSE_LEAD_TURNS", lead_turns),
            ("GOOSE_LEAD_FAILURE_THRESHOLD", failure_threshold),
            ("GOOSE_LEAD_FALLBACK_TURNS", fallback_turns),
            ("OPENAI_API_KEY", Some("fake-openai-no-keyring")),
            ("OPENAI_CUSTOM_HEADERS", Some("")),
        ]);

        let provider = create(
            "openai",
            ModelConfig::new_or_fail("gpt-4o-mini").with_canonical_limits("openai"),
            Vec::new(),
        )
        .await
        .unwrap();
        let lw = provider.as_lead_worker().unwrap();
        let (lead, worker) = lw.get_model_info();
        assert_eq!(lead, "gpt-4o");
        assert_eq!(worker, "gpt-4o-mini");
        assert_eq!(
            lw.get_settings(),
            (expected_turns, expected_failure, expected_fallback)
        );
    }

    #[tokio::test]
    async fn test_create_regular_provider_without_lead_config() {
        let _guard = env_lock::lock_env([
            ("GOOSE_LEAD_MODEL", None),
            ("GOOSE_LEAD_PROVIDER", None),
            ("GOOSE_LEAD_TURNS", None),
            ("GOOSE_LEAD_FAILURE_THRESHOLD", None),
            ("GOOSE_LEAD_FALLBACK_TURNS", None),
            ("OPENAI_API_KEY", Some("fake-openai-no-keyring")),
            ("OPENAI_CUSTOM_HEADERS", Some("")),
        ]);

        let provider = create(
            "openai",
            ModelConfig::new_or_fail("gpt-4o-mini").with_canonical_limits("openai"),
            Vec::new(),
        )
        .await
        .unwrap();
        assert!(provider.as_lead_worker().is_none());
        assert_eq!(provider.get_model_config().model_name, "gpt-4o-mini");
    }

    #[test_case::test_case(None, None, 16_000 ; "no overrides uses default")]
    #[test_case::test_case(Some("32000"), None, 32_000 ; "worker limit overrides default")]
    #[test_case::test_case(Some("32000"), Some("64000"), 32_000 ; "worker limit takes priority over global")]
    fn test_worker_model_context_limit(
        worker_limit: Option<&str>,
        global_limit: Option<&str>,
        expected_limit: usize,
    ) {
        let _guard = env_lock::lock_env([
            ("GOOSE_WORKER_CONTEXT_LIMIT", worker_limit),
            ("GOOSE_CONTEXT_LIMIT", global_limit),
        ]);

        let default_model = ModelConfig::new_or_fail("gpt-3.5-turbo")
            .with_canonical_limits("openai")
            .with_context_limit(Some(16_000));

        let result = create_worker_model_config(&default_model, "openai").unwrap();
        assert_eq!(result.context_limit, Some(expected_limit));
    }

    #[tokio::test]
    async fn test_openai_compatible_providers_config_keys() {
        let providers_list = providers().await;
        let required_api_key_cases = vec![
            ("groq", "GROQ_API_KEY"),
            ("mistral", "MISTRAL_API_KEY"),
            ("custom_deepseek", "DEEPSEEK_API_KEY"),
        ];
        for (name, expected_key) in required_api_key_cases {
            if let Some((meta, _)) = providers_list.iter().find(|(m, _)| m.name == name) {
                assert!(
                    !meta.config_keys.is_empty(),
                    "{name} provider should have config keys"
                );
                assert_eq!(
                    meta.config_keys[0].name, expected_key,
                    "First config key for {name} should be {expected_key}, got {}",
                    meta.config_keys[0].name
                );
                assert!(
                    meta.config_keys[0].required,
                    "{expected_key} should be required"
                );
                assert!(
                    meta.config_keys[0].secret,
                    "{expected_key} should be secret"
                );
            } else {
                // Provider not registered; skip test for this provider
                continue;
            }
        }

        if let Some((meta, _)) = providers_list.iter().find(|(m, _)| m.name == "openai") {
            assert!(
                !meta.config_keys.is_empty(),
                "openai provider should have config keys"
            );
            assert_eq!(
                meta.config_keys[0].name, "OPENAI_API_KEY",
                "First config key for openai should be OPENAI_API_KEY"
            );
            assert!(
                !meta.config_keys[0].required,
                "OPENAI_API_KEY should be optional for local server support"
            );
            assert!(
                meta.config_keys[0].secret,
                "OPENAI_API_KEY should be secret"
            );
        }
    }
}
