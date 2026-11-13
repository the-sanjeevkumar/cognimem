"""
CogniMem v2.0 - Abstract Memory Backend Interfaces
Defines the contract for pluggable memory storage backends.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class MemoryRecord(BaseModel):
    """Universal memory record format."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    content: str
    memory_type: str  # "working", "episodic", "semantic", "archival"
    tier: str = "episodic"  # "immediate", "short_term", "long_term", "archival"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    importance_score: float = 1.0
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    parent_id: Optional[str] = None
    compressed: bool = False
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None


class MemoryQuery(BaseModel):
    """Structured memory query."""
    query: str
    agent_id: str
    top_k: int = 5
    filters: Dict[str, Any] = Field(default_factory=dict)
    memory_types: Optional[List[str]] = None
    tiers: Optional[List[str]] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    min_importance: float = 0.0
    include_embeddings: bool = False


class SearchResult(BaseModel):
    """Search result with scoring."""
    record: MemoryRecord
    similarity_score: float
    importance_score: float
    recency_score: float
    final_score: float
    matched_fields: List[str] = Field(default_factory=list)


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend connection."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the backend connection."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        pass
    
    # CRUD operations
    @abstractmethod
    async def add(self, record: MemoryRecord) -> str:
        """Add a memory record. Returns the record ID."""
        pass
    
    @abstractmethod
    async def add_batch(self, records: List[MemoryRecord]) -> List[str]:
        """Add multiple memory records. Returns list of record IDs."""
        pass
    
    @abstractmethod
    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Get a memory record by ID."""
        pass
    
    @abstractmethod
    async def update(self, record: MemoryRecord) -> bool:
        """Update a memory record. Returns True if successful."""
        pass
    
    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory record. Returns True if successful."""
        pass
    
    @abstractmethod
    async def delete_by_agent(self, agent_id: str, memory_type: Optional[str] = None) -> int:
        """Delete all memories for an agent. Returns count deleted."""
        pass
    
    # Query operations
    @abstractmethod
    async def search(self, query: MemoryQuery) -> List[SearchResult]:
        """Search memories based on query."""
        pass
    
    @abstractmethod
    async def search_by_embedding(
        self, 
        embedding: List[float], 
        agent_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search by vector embedding."""
        pass
    
    @abstractmethod
    async def get_by_tags(
        self, 
        agent_id: str, 
        tags: List[str], 
        match_all: bool = False,
        limit: int = 100
    ) -> List[MemoryRecord]:
        """Get memories by tags."""
        pass
    
    @abstractmethod
    async def get_recent(
        self, 
        agent_id: str, 
        limit: int = 10,
        memory_type: Optional[str] = None,
        tier: Optional[str] = None
    ) -> List[MemoryRecord]:
        """Get recent memories for an agent."""
        pass
    
    @abstractmethod
    async def get_by_time_range(
        self,
        agent_id: str,
        start: datetime,
        end: datetime,
        limit: int = 100
    ) -> List[MemoryRecord]:
        """Get memories within a time range."""
        pass
    
    # Tier operations
    @abstractmethod
    async def promote_to_tier(self, memory_id: str, target_tier: str) -> bool:
        """Promote a memory to a higher tier."""
        pass
    
    @abstractmethod
    async def demote_to_tier(self, memory_id: str, target_tier: str) -> bool:
        """Demote a memory to a lower tier."""
        pass
    
    @abstractmethod
    async def get_tier_stats(self, agent_id: str) -> Dict[str, int]:
        """Get memory count per tier for an agent."""
        pass
    
    # Analytics
    @abstractmethod
    async def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for an agent."""
        pass
    
    @abstractmethod
    async def get_importance_distribution(self, agent_id: str) -> Dict[str, int]:
        """Get distribution of importance scores."""
        pass


class VectorBackend(ABC):
    """Abstract base class for vector similarity search backends."""
    
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass
    
    @abstractmethod
    async def add_vectors(
        self, 
        ids: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]]
    ) -> None:
        pass
    
    @abstractmethod
    async def search_vectors(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> None:
        pass
    
    @abstractmethod
    async def update_vectors(
        self, 
        ids: List[str], 
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        pass


class GraphBackend(ABC):
    """Abstract base class for knowledge graph backends."""
    
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass
    
    @abstractmethod
    async def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    async def add_edge(self, source: str, target: str, weight: float = 1.0, properties: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    @abstractmethod
    async def get_neighbors(self, node_id: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def find_paths(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        pass
    
    @abstractmethod
    async def get_subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def query_by_properties(self, properties: Dict[str, Any], limit: int = 100) -> List[str]:
        pass
    
    @abstractmethod
    async def compute_centrality(self, node_ids: Optional[List[str]] = None) -> Dict[str, float]:
        pass
    
    @abstractmethod
    async def cluster_nodes(self, algorithm: str = "louvain") -> Dict[str, int]:
        pass


class EmbeddingBackend(ABC):
    """Abstract base class for embedding generation backends."""
    
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass