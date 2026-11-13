"""
CogniMem v2.0 - Main SDK
Unified interface for the advanced cognitive memory system.
"""

import asyncio
from typing import List, Optional, Dict, Any, AsyncIterator
from datetime import datetime
from contextlib import asynccontextmanager

from cognimem.config import get_settings, CogniMemSettings, configure_settings
from cognimem.backends.base import MemoryBackend, MemoryRecord, MemoryQuery, SearchResult
from cognimem.backends.local import LocalMemoryBackend
from cognimem.core.hierarchical import (
    HierarchicalMemory, MemoryTier, create_hierarchical_memory
)
from cognimem.core.compression import MemoryCompressor, create_memory_compressor
from cognimem.core.query_planner import QueryPlanner, create_query_planner
from cognimem.core.proactive import ProactiveMemoryManager, create_proactive_manager
from cognimem.core.multiagent import MultiAgentMemoryManager, create_multiagent_manager
from cognimem.core.sessions import SessionManager, CrossSessionRetriever, create_session_manager


class CogniMem:
    """
    CogniMem v2.0 - Advanced Cognitive Memory for AI Agents
    
    Features:
    - Hierarchical memory tiers (immediate, short-term, long-term, archival)
    - Automatic memory consolidation with Ebbinghaus forgetting curve
    - LLM-based memory compression and summarization
    - Intelligent query planning and decomposition
    - Proactive/anticipatory memory retrieval
    - Multi-agent shared memory with access control
    - Cross-session continuity and context bridging
    - Async-first architecture with pluggable backends
    """
    
    def __init__(
        self,
        agent_id: str = "default_agent",
        settings: Optional[CogniMemSettings] = None,
        backend: Optional[MemoryBackend] = None,
        auto_initialize: bool = True
    ):
        self.agent_id = agent_id
        self.settings = settings or get_settings()
        self.settings.agent_id = agent_id
        
        # Core components (lazy initialized)
        self._backend = backend
        self._hierarchical: Optional[HierarchicalMemory] = None
        self._compressor: Optional[MemoryCompressor] = None
        self._query_planner: Optional[QueryPlanner] = None
        self._proactive: Optional[ProactiveMemoryManager] = None
        self._multiagent: Optional[MultiAgentMemoryManager] = None
        self._session_manager: Optional[SessionManager] = None
        self._cross_session: Optional[CrossSessionRetriever] = None
        self._current_session_id: Optional[str] = None
        
        self._initialized = False
        self._auto_initialize = auto_initialize
    
    @property
    def backend(self) -> MemoryBackend:
        if self._backend is None:
            self._backend = LocalMemoryBackend(self.settings)
        return self._backend
    
    @property
    def hierarchical(self) -> HierarchicalMemory:
        if self._hierarchical is None:
            self._hierarchical = HierarchicalMemory(self.backend, self.settings)
        return self._hierarchical
    
    @property
    def compressor(self) -> MemoryCompressor:
        if self._compressor is None:
            self._compressor = MemoryCompressor(self.backend, self.settings)
        return self._compressor
    
    @property
    def query_planner(self) -> QueryPlanner:
        if self._query_planner is None:
            self._query_planner = QueryPlanner(
                self._create_query_analyzer(), 
                self.settings
            )
        return self._query_planner
    
    def _create_query_analyzer(self):
        from cognimem.core.query_planner import LLMQueryAnalyzer
        return LLMQueryAnalyzer(self.settings)
    
    @property
    def proactive(self) -> ProactiveMemoryManager:
        if self._proactive is None:
            self._proactive = ProactiveMemoryManager(
                self.backend, 
                self.query_planner, 
                self.settings
            )
        return self._proactive
    
    @property
    def multiagent(self) -> MultiAgentMemoryManager:
        if self._multiagent is None:
            self._multiagent = MultiAgentMemoryManager(self.backend, self.settings)
        return self._multiagent
    
    @property
    def session_manager(self) -> SessionManager:
        if self._session_manager is None:
            self._session_manager = SessionManager(self.backend, self.compressor, self.settings)
        return self._session_manager
    
    @property
    def cross_session(self) -> CrossSessionRetriever:
        if self._cross_session is None:
            self._cross_session = CrossSessionRetriever(self.session_manager, self.backend)
        return self._cross_session
    
    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        await self.backend.initialize()
        await self.hierarchical.initialize()
        await self.session_manager.initialize()
        
        # Start proactive background prefetch
        await self.proactive.start_background_prefetch()
        
        self._initialized = True
    
    async def close(self) -> None:
        """Cleanup resources."""
        await self.proactive.stop_background_prefetch()
        await self.backend.close()
        self._initialized = False
    
    # --- Core Memory Operations ---
    
    async def remember(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        tier: Optional[MemoryTier] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Store a memory with automatic tier placement.
        
        Args:
            content: The memory content
            importance: Importance score (0.0-1.0)
            tags: Optional tags for categorization
            tier: Explicit tier override
            metadata: Additional metadata
            session_id: Associate with a session
            
        Returns:
            Memory ID
        """
        await self._ensure_initialized()
        
        tags = tags or []
        metadata = metadata or {}
        
        if session_id:
            metadata["session_id"] = session_id
            tags.append(f"session:{session_id}")
        elif self._current_session_id:
            metadata["session_id"] = self._current_session_id
            tags.append(f"session:{self._current_session_id}")
        
        return await self.hierarchical.remember(
            content=content,
            agent_id=self.agent_id,
            importance=importance,
            tags=tags,
            tier=tier,
            metadata=metadata
        )
    
    async def recall(
        self,
        query: str,
        top_k: int = 5,
        tiers: Optional[List[MemoryTier]] = None,
        min_importance: float = 0.0,
        use_query_planner: bool = True,
        include_cross_session: bool = True
    ) -> List[SearchResult]:
        """
        Recall memories with intelligent retrieval.
        
        Args:
            query: Search query
            top_k: Number of results
            tiers: Specific tiers to search
            min_importance: Minimum importance threshold
            use_query_planner: Enable query planning for complex queries
            include_cross_session: Include cross-session context
            
        Returns:
            List of search results ranked by relevance
        """
        await self._ensure_initialized()
        
        if use_query_planner and self.settings.query_planner.enabled:
            # Use query planner for intelligent retrieval
            plan = await self.query_planner.create_plan(query, self.agent_id)
            results = await self.query_planner.execute_plan(plan, self.backend, self.agent_id)
        else:
            # Simple hierarchical recall
            results = await self.hierarchical.recall(
                query=query,
                agent_id=self.agent_id,
                top_k=top_k,
                tiers=tiers,
                min_importance=min_importance
            )
        
        # Add cross-session context
        if include_cross_session and self._current_session_id:
            cross_results = await self.cross_session.recall_with_context(
                query=query,
                agent_id=self.agent_id,
                current_session_id=self._current_session_id,
                top_k=top_k
            )
            # Merge and deduplicate
            results = self._merge_results(results, cross_results, top_k)
        
        return results[:top_k]
    
    def _merge_results(self, primary: List[SearchResult], secondary: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Merge and deduplicate results from multiple sources."""
        seen = set()
        merged = []
        
        # Primary results get priority
        for r in primary:
            if r.record.id not in seen:
                seen.add(r.record.id)
                r.record.metadata["source"] = r.record.metadata.get("source", "primary")
                merged.append(r)
        
        # Add secondary results
        for r in secondary:
            if r.record.id not in seen:
                seen.add(r.record.id)
                merged.append(r)
        
        merged.sort(key=lambda x: x.final_score, reverse=True)
        return merged
    
    async def recall_simple(self, query: str, top_k: int = 3) -> str:
        """Simple recall returning formatted string (backward compatible)."""
        results = await self.recall(query, top_k=top_k)
        if not results:
            return "No relevant memories found."
        
        return "\n".join([
            f"[{r.record.timestamp.strftime('%Y-%m-%d %H:%M')}] (Relevance: {r.final_score:.2f}): {r.record.content}"
            for r in results
        ])
    
    # --- Session Management ---
    
    async def start_session(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        parent_session_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Start a new memory session."""
        await self._ensure_initialized()
        
        session = await self.session_manager.create_session(
            agent_id=self.agent_id,
            metadata=metadata,
            parent_session_id=parent_session_id,
            tags=tags
        )
        
        self._current_session_id = session.session_id
        return session.session_id
    
    async def end_session(self, create_summary: bool = True) -> bool:
        """End the current session."""
        if not self._current_session_id:
            return False
        
        result = await self.session_manager.end_session(
            self._current_session_id,
            create_summary=create_summary
        )
        
        self._current_session_id = None
        return result
    
    async def get_current_session(self) -> Optional[str]:
        """Get current session ID."""
        return self._current_session_id
    
    async def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List sessions for this agent."""
        await self._ensure_initialized()
        
        from cognimem.core.sessions import SessionStatus
        session_status = SessionStatus(status) if status else None
        
        sessions = await self.session_manager.list_agent_sessions(
            self.agent_id,
            status=session_status,
            limit=limit
        )
        
        return [
            {
                "session_id": s.session_id,
                "status": s.status.value,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "memory_count": s.memory_count,
                "summary": s.summary,
                "tags": s.tags,
            }
            for s in sessions
        ]
    
    # --- Advanced Operations ---
    
    async def promote_memory(self, memory_id: str, target_tier: MemoryTier) -> bool:
        """Promote a memory to a higher tier."""
        await self._ensure_initialized()
        return await self.hierarchical.promote_memory(memory_id, target_tier)
    
    async def demote_memory(self, memory_id: str, target_tier: MemoryTier) -> bool:
        """Demote a memory to a lower tier."""
        await self._ensure_initialized()
        return await self.hierarchical.demote_memory(memory_id, target_tier)
    
    async def compress_memory(self, memory_id: str, level: str = "medium") -> Optional[str]:
        """Compress a memory record."""
        await self._ensure_initialized()
        
        from cognimem.core.compression import CompressionLevel
        comp_level = CompressionLevel(level)
        
        record = await self.backend.get(memory_id)
        if not record:
            return None
        
        compressed = await self.compressor.compress_memory(record, comp_level)
        return await self.backend.add(compressed)
    
    async def auto_compress_old_memories(
        self,
        age_threshold_days: int = 30,
        level: str = "aggressive"
    ) -> int:
        """Automatically compress old memories."""
        await self._ensure_initialized()
        
        from cognimem.core.compression import CompressionLevel
        comp_level = CompressionLevel(level)
        
        return await self.compressor.auto_compress_old_memories(
            self.agent_id,
            age_threshold_days,
            comp_level
        )
    
    # --- Multi-Agent Operations ---
    
    async def create_shared_space(
        self,
        name: str,
        namespace: str = "team",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a shared memory space."""
        await self._ensure_initialized()
        
        from cognimem.core.multiagent import MemoryNamespace
        ns = MemoryNamespace(namespace)
        
        space = await self.multiagent.create_space(
            name=name,
            owner_id=self.agent_id,
            namespace=ns,
            metadata=metadata
        )
        
        return {
            "space_id": space.space_id,
            "name": space.name,
            "namespace": space.namespace.value,
            "owner_id": space.owner_id,
        }
    
    async def remember_shared(
        self,
        space_id: str,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """Store a memory in a shared space."""
        await self._ensure_initialized()
        
        return await self.multiagent.remember_shared(
            space_id=space_id,
            agent_id=self.agent_id,
            content=content,
            importance=importance,
            tags=tags
        )
    
    async def recall_shared(
        self,
        space_id: str,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Recall memories from a shared space."""
        await self._ensure_initialized()
        
        return await self.multiagent.recall_shared(
            space_id=space_id,
            agent_id=self.agent_id,
            query=query,
            top_k=top_k
        )
    
    # --- Proactive Memory ---
    
    async def on_context_change(self, context: Dict[str, Any], max_results: int = 3) -> List[SearchResult]:
        """Notify proactive system of context change."""
        await self._ensure_initialized()
        
        result = await self.proactive.on_context_change(
            agent_id=self.agent_id,
            context=context,
            max_results=max_results
        )
        
        return result.memories
    
    async def on_entity_mention(self, entity: str, context: Dict[str, Any], max_results: int = 3) -> List[SearchResult]:
        """Notify proactive system of entity mention."""
        await self._ensure_initialized()
        
        result = await self.proactive.on_entity_mention(
            agent_id=self.agent_id,
            entity=entity,
            context=context,
            max_results=max_results
        )
        
        return result.memories
    
    async def on_decision_pending(self, decision: str, context: Dict[str, Any], max_results: int = 5) -> List[SearchResult]:
        """Notify proactive system of pending decision."""
        await self._ensure_initialized()
        
        result = await self.proactive.on_decision_pending(
            agent_id=self.agent_id,
            decision=decision,
            context=context,
            max_results=max_results
        )
        
        return result.memories
    
    # --- Statistics & Monitoring ---
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        await self._ensure_initialized()
        
        base_stats = await self.backend.get_stats(self.agent_id)
        tier_stats = await self.hierarchical.get_stats(self.agent_id)
        session_stats = await self.session_manager.get_session_stats(self.agent_id)
        
        return {
            "agent_id": self.agent_id,
            "base": base_stats,
            "tiers": tier_stats.get("tiers", {}),
            "sessions": session_stats,
        }
    
    async def get_tier_stats(self) -> Dict[str, Any]:
        """Get tier-specific statistics."""
        await self._ensure_initialized()
        return await self.hierarchical.get_stats(self.agent_id)
    
    # --- Maintenance ---
    
    async def rebalance_tiers(self) -> Dict[str, int]:
        """Rebalance memories across tiers."""
        await self._ensure_initialized()
        return await self.hierarchical.rebalance(self.agent_id)
    
    async def cleanup_expired(self) -> int:
        """Clean up expired memories."""
        await self._ensure_initialized()
        return await self.hierarchical.cleanup(self.agent_id)
    
    async def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Clean up old sessions."""
        await self._ensure_initialized()
        return await self.session_manager.cleanup_old_sessions(self.agent_id, max_age_days)
    
    # --- Internal ---
    
    async def _ensure_initialized(self) -> None:
        if not self._initialized and self._auto_initialize:
            await self.initialize()
    
    async def __aenter__(self) -> "CogniMem":
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


# --- Backward Compatibility ---

class CogniMemSync:
    """Synchronous wrapper for backward compatibility."""
    
    def __init__(self, agent_id: str = "default_agent", **kwargs):
        self._async = CogniMem(agent_id=agent_id, **kwargs)
        self._loop = None
    
    def _get_loop(self):
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def _run(self, coro):
        loop = self._get_loop()
        return loop.run_until_complete(coro)
    
    def remember(self, content: str, importance: float = 0.5, tags: Optional[List[str]] = None) -> str:
        return self._run(self._async.remember(content, importance, tags))
    
    def recall(self, query: str, top_k: int = 3) -> str:
        return self._run(self._async.recall_simple(query, top_k))
    
    def initialize(self) -> None:
        self._run(self._async.initialize())
    
    def close(self) -> None:
        self._run(self._async.close())


# --- Factory Functions ---

async def create_cognimem(
    agent_id: str = "default_agent",
    settings: Optional[CogniMemSettings] = None,
    backend: Optional[MemoryBackend] = None
) -> CogniMem:
    """Create and initialize a CogniMem instance."""
    mem = CogniMem(agent_id=agent_id, settings=settings, backend=backend)
    await mem.initialize()
    return mem


def create_cognimem_sync(
    agent_id: str = "default_agent",
    **kwargs
) -> CogniMemSync:
    """Create a synchronous CogniMem instance."""
    return CogniMemSync(agent_id=agent_id, **kwargs)


# --- Version ---
__version__ = "2.0.0"