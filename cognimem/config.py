"""
CogniMem v2.0 - Advanced Configuration System
Pydantic Settings-based configuration with validation and environment overrides.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    model: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None  # auto-detect
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_dir: Optional[str] = None


class WorkingMemoryConfig(BaseSettings):
    """Working memory (immediate/short-term) configuration."""
    limit: int = 10
    ttl_seconds: int = 3600  # 1 hour default TTL
    auto_flush: bool = True


class EpisodicMemoryConfig(BaseSettings):
    """Episodic memory (vector-based experiences) configuration."""
    persist_directory: str = "./chroma_db"
    collection_name: str = "episodic_memories"
    distance_metric: str = "cosine"  # cosine, l2, ip
    hnsw_space: str = "cosine"
    hnsw_construction_ef: int = 100
    hnsw_search_ef: int = 50
    hnsw_m: int = 16


class SemanticMemoryConfig(BaseSettings):
    """Semantic memory (knowledge graph) configuration."""
    persist_directory: str = "./semantic_graphs"
    graph_format: str = "json"  # json, graphml, gexf
    auto_save_interval_seconds: int = 300
    max_nodes_per_concept: int = 1000


class ConsolidationConfig(BaseSettings):
    """Memory consolidation (sleep cycle) configuration."""
    enabled: bool = True
    interval_hours: int = 6
    decay_constant_k: float = 24.0
    promotion_threshold: float = 0.7
    pruning_threshold: float = 0.1
    min_importance_for_promotion: float = 0.5
    max_promotions_per_cycle: int = 100
    enable_background_scheduler: bool = True


class RetrievalConfig(BaseSettings):
    """Hybrid retrieval configuration."""
    vector_weight: float = 0.6
    importance_weight: float = 0.3
    recency_weight: float = 0.1
    top_k: int = 5
    max_candidates: int = 50
    rerank_enabled: bool = False
    rerank_model: Optional[str] = None
    diversity_penalty: float = 0.1
    min_similarity_threshold: float = 0.3


class CompressionConfig(BaseSettings):
    """Memory compression/summarization configuration."""
    enabled: bool = True
    llm_model: str = "gpt-4o-mini"
    max_tokens_per_summary: int = 500
    compression_ratio_target: float = 0.3
    preserve_entities: bool = True
    preserve_decisions: bool = True
    preserve_commitments: bool = True
    hierarchical_levels: List[str] = ["turn", "session", "day", "week"]


class QueryPlannerConfig(BaseSettings):
    """Query planning and decomposition configuration."""
    enabled: bool = True
    llm_model: str = "gpt-4o-mini"
    max_sub_queries: int = 5
    parallel_retrieval: bool = True
    fusion_strategy: str = "rrf"  # rrf, weighted, max
    query_rewrite_enabled: bool = True


class MultiAgentConfig(BaseSettings):
    """Multi-agent shared memory configuration."""
    enabled: bool = False
    namespace_prefix: str = "team_"
    default_permissions: str = "read_write"  # read, write, read_write, admin
    conflict_resolution: str = "last_write_wins"  # last_write_wins, merge, manual
    audit_log_enabled: bool = True


class StorageConfig(BaseSettings):
    """Storage backend configuration."""
    backend: str = "local"  # local, redis, postgresql, s3
    redis_url: Optional[str] = None
    postgresql_url: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    s3_prefix: str = "cognimem/"


class APIConfig(BaseSettings):
    """REST API configuration."""
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = ["*"]
    rate_limit: int = 100


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    file: Optional[str] = None
    max_bytes: int = 10_000_000
    backup_count: int = 5


class CogniMemSettings(BaseSettings):
    """Main CogniMem settings - loads from config.yaml and environment variables."""
    
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_prefix="COGNIMEM_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Core settings
    agent_id: str = "default_agent"
    version: str = "2.0.0"
    
    # Component configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    working_memory: WorkingMemoryConfig = Field(default_factory=WorkingMemoryConfig)
    episodic_memory: EpisodicMemoryConfig = Field(default_factory=EpisodicMemoryConfig)
    semantic_memory: SemanticMemoryConfig = Field(default_factory=SemanticMemoryConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    query_planner: QueryPlannerConfig = Field(default_factory=QueryPlannerConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("agent_id cannot be empty")
        return v.strip()
    
    def get_chroma_path(self) -> str:
        """Get the ChromaDB path for this agent."""
        import os
        return os.path.join(self.episodic_memory.persist_directory, self.agent_id)
    
    def get_semantic_path(self) -> str:
        """Get the semantic graph path for this agent."""
        import os
        return os.path.join(self.semantic_memory.persist_directory, f"{self.agent_id}.{self.semantic_memory.graph_format}")


# Global settings instance
_settings: Optional[CogniMemSettings] = None


def get_settings() -> CogniMemSettings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = CogniMemSettings()
    return _settings


def configure_settings(**kwargs) -> CogniMemSettings:
    """Configure global settings programmatically."""
    global _settings
    _settings = CogniMemSettings(**kwargs)
    return _settings


def reset_settings() -> None:
    """Reset global settings (useful for testing)."""
    global _settings
    _settings = None