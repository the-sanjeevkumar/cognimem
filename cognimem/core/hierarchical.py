"""
CogniMem v2.0 - Hierarchical Memory Tiers
Implements multi-tier memory architecture with automatic promotion/demotion.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import heapq

from cognimem.config import get_settings
from cognimem.backends.base import MemoryBackend, MemoryRecord, MemoryQuery, SearchResult
from cognimem.backends.local import LocalMemoryBackend


class MemoryTier(str, Enum):
    """Memory tier levels."""
    IMMEDIATE = "immediate"      # Working memory - milliseconds to minutes
    SHORT_TERM = "short_term"    # Episodic memory - minutes to hours
    LONG_TERM = "long_term"      # Semantic memory - hours to years
    ARCHIVAL = "archival"        # Cold storage - years


@dataclass
class TierConfig:
    """Configuration for a memory tier."""
    name: MemoryTier
    capacity: int
    ttl_seconds: int
    min_importance: float
    retrieval_latency_ms: int  # Target latency
    storage_backend: str = "local"
    auto_promote: bool = True
    auto_demote: bool = True
    compression_enabled: bool = False


DEFAULT_TIER_CONFIGS = {
    MemoryTier.IMMEDIATE: TierConfig(
        name=MemoryTier.IMMEDIATE,
        capacity=10,
        ttl_seconds=300,  # 5 minutes
        min_importance=0.0,
        retrieval_latency_ms=10,
        compression_enabled=False,
    ),
    MemoryTier.SHORT_TERM: TierConfig(
        name=MemoryTier.SHORT_TERM,
        capacity=1000,
        ttl_seconds=86400 * 7,  # 1 week
        min_importance=0.1,
        retrieval_latency_ms=50,
        compression_enabled=False,
    ),
    MemoryTier.LONG_TERM: TierConfig(
        name=MemoryTier.LONG_TERM,
        capacity=100000,
        ttl_seconds=86400 * 365,  # 1 year
        min_importance=0.5,
        retrieval_latency_ms=100,
        compression_enabled=True,
    ),
    MemoryTier.ARCHIVAL: TierConfig(
        name=MemoryTier.ARCHIVAL,
        capacity=1000000,
        ttl_seconds=86400 * 365 * 10,  # 10 years
        min_importance=0.7,
        retrieval_latency_ms=500,
        compression_enabled=True,
    ),
}


class TierManager:
    """Manages memory tiers and cross-tier operations."""
    
    def __init__(self, backend: MemoryBackend, settings=None):
        self.backend = backend
        self.settings = settings or get_settings()
        self.tier_configs = DEFAULT_TIER_CONFIGS.copy()
        self._tier_queues: Dict[MemoryTier, List[tuple]] = {
            tier: [] for tier in MemoryTier
        }
        self._promotion_callbacks: List[Callable] = []
        self._demotion_callbacks: List[Callable] = []
    
    def configure_tier(self, tier: MemoryTier, config: TierConfig) -> None:
        """Update tier configuration."""
        self.tier_configs[tier] = config
    
    def register_promotion_callback(self, callback: Callable) -> None:
        """Register callback for memory promotion events."""
        self._promotion_callbacks.append(callback)
    
    def register_demotion_callback(self, callback: Callable) -> None:
        """Register callback for memory demotion events."""
        self._demotion_callbacks.append(callback)
    
    async def add_to_tier(self, record: MemoryRecord, tier: MemoryTier) -> str:
        """Add a memory record to a specific tier."""
        record.tier = tier.value
        record.memory_type = self._tier_to_memory_type(tier)
        return await self.backend.add(record)
    
    def _tier_to_memory_type(self, tier: MemoryTier) -> str:
        mapping = {
            MemoryTier.IMMEDIATE: "working",
            MemoryTier.SHORT_TERM: "episodic",
            MemoryTier.LONG_TERM: "semantic",
            MemoryTier.ARCHIVAL: "archival",
        }
        return mapping.get(tier, "episodic")
    
    async def promote(self, memory_id: str, target_tier: MemoryTier) -> bool:
        """Promote a memory to a higher tier."""
        success = await self.backend.promote_to_tier(memory_id, target_tier.value)
        if success:
            for callback in self._promotion_callbacks:
                try:
                    await callback(memory_id, target_tier)
                except Exception:
                    pass
        return success
    
    async def demote(self, memory_id: str, target_tier: MemoryTier) -> bool:
        """Demote a memory to a lower tier."""
        success = await self.backend.demote_to_tier(memory_id, target_tier.value)
        if success:
            for callback in self._demotion_callbacks:
                try:
                    await callback(memory_id, target_tier)
                except Exception:
                    pass
        return success
    
    async def auto_tier_memory(self, record: MemoryRecord) -> MemoryTier:
        """Determine the appropriate tier for a memory based on importance and age."""
        importance = record.importance_score
        age_hours = (datetime.utcnow() - record.timestamp).total_seconds() / 3600
        
        # High importance memories go to long-term
        if importance >= self.tier_configs[MemoryTier.LONG_TERM].min_importance:
            return MemoryTier.LONG_TERM
        
        # Medium importance with some age goes to short-term
        if importance >= self.tier_configs[MemoryTier.SHORT_TERM].min_importance:
            return MemoryTier.SHORT_TERM
        
        # Recent memories stay in immediate
        if age_hours < 1:
            return MemoryTier.IMMEDIATE
        
        return MemoryTier.SHORT_TERM
    
    async def rebalance_tiers(self, agent_id: str) -> Dict[str, int]:
        """Rebalance memories across tiers based on current state."""
        stats = await self.backend.get_tier_stats(agent_id)
        moved = {tier.value: 0 for tier in MemoryTier}
        
        # Get all memories for the agent
        for tier_name in MemoryTier:
            tier = MemoryTier(tier_name)
            config = self.tier_configs[tier]
            
            # Check if tier is over capacity
            current_count = stats.get(tier.value, 0)
            if current_count > config.capacity:
                # Need to demote oldest/least important
                memories = await self.backend.get_recent(
                    agent_id, 
                    limit=current_count,
                    tier=tier.value
                )
                
                # Sort by importance (ascending) then age (descending)
                memories.sort(key=lambda m: (m.importance_score, -m.timestamp.timestamp()))
                
                # Demote excess
                excess = current_count - config.capacity
                for memory in memories[:excess]:
                    # Determine target tier
                    target = self._get_demotion_target(memory)
                    if target != tier:
                        await self.demote(memory.id, target)
                        moved[tier.value] += 1
        
        return moved
    
    def _get_demotion_target(self, record: MemoryRecord) -> MemoryTier:
        """Determine where to demote a memory."""
        current_tier = MemoryTier(record.tier)
        
        if current_tier == MemoryTier.ARCHIVAL:
            return MemoryTier.LONG_TERM
        elif current_tier == MemoryTier.LONG_TERM:
            return MemoryTier.SHORT_TERM
        elif current_tier == MemoryTier.SHORT_TERM:
            return MemoryTier.IMMEDIATE
        else:
            return MemoryTier.IMMEDIATE
    
    async def cleanup_expired(self, agent_id: str) -> int:
        """Remove expired memories from all tiers."""
        total_removed = 0
        
        for tier in MemoryTier:
            config = self.tier_configs[tier]
            cutoff = datetime.utcnow() - timedelta(seconds=config.ttl_seconds)
            
            # Get memories in this tier
            memories = await self.backend.get_recent(
                agent_id,
                limit=10000,
                tier=tier.value
            )
            
            for memory in memories:
                if memory.timestamp < cutoff and memory.importance_score < config.min_importance:
                    await self.backend.delete(memory.id)
                    total_removed += 1
        
        return total_removed
    
    async def get_tier_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed statistics for all tiers."""
        base_stats = await self.backend.get_tier_stats(agent_id)
        
        detailed = {}
        for tier in MemoryTier:
            config = self.tier_configs[tier]
            count = base_stats.get(tier.value, 0)
            detailed[tier.value] = {
                "count": count,
                "capacity": config.capacity,
                "utilization": count / config.capacity if config.capacity > 0 else 0,
                "ttl_hours": config.ttl_seconds / 3600,
                "min_importance": config.min_importance,
                "target_latency_ms": config.retrieval_latency_ms,
            }
        
        return detailed


class HierarchicalMemory:
    """Main hierarchical memory interface."""
    
    def __init__(self, backend: MemoryBackend, settings=None):
        self.backend = backend
        self.settings = settings or get_settings()
        self.tier_manager = TierManager(backend, settings)
        self._initialized = False
    
    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.backend.initialize()
        self._initialized = True
    
    async def close(self) -> None:
        await self.backend.close()
        self._initialized = False
    
    async def remember(
        self,
        content: str,
        agent_id: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        tier: Optional[MemoryTier] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a memory, automatically determining the appropriate tier."""
        await self.initialize()
        
        tags = tags or []
        metadata = metadata or {}
        
        # Create memory record
        record = MemoryRecord(
            agent_id=agent_id,
            content=content,
            importance_score=importance,
            tags=tags,
            metadata=metadata,
        )
        
        # Determine tier if not specified
        if tier is None:
            tier = await self.tier_manager.auto_tier_memory(record)
        
        return await self.tier_manager.add_to_tier(record, tier)
    
    async def recall(
        self,
        query: str,
        agent_id: str,
        top_k: int = 5,
        tiers: Optional[List[MemoryTier]] = None,
        min_importance: float = 0.0,
    ) -> List[SearchResult]:
        """Recall memories across tiers."""
        await self.initialize()
        
        memory_query = MemoryQuery(
            query=query,
            agent_id=agent_id,
            top_k=top_k,
            tiers=[t.value for t in tiers] if tiers else None,
            min_importance=min_importance,
        )
        
        return await self.backend.search(memory_query)
    
    async def recall_by_tier(
        self,
        query: str,
        agent_id: str,
        tier: MemoryTier,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Recall memories from a specific tier."""
        return await self.recall(query, agent_id, top_k, tiers=[tier])
    
    async def get_recent(
        self,
        agent_id: str,
        limit: int = 10,
        tier: Optional[MemoryTier] = None,
    ) -> List[MemoryRecord]:
        """Get recent memories from a tier or all tiers."""
        await self.initialize()
        return await self.backend.get_recent(
            agent_id,
            limit=limit,
            tier=tier.value if tier else None
        )
    
    async def promote_memory(self, memory_id: str, target_tier: MemoryTier) -> bool:
        """Manually promote a memory to a higher tier."""
        return await self.tier_manager.promote(memory_id, target_tier)
    
    async def demote_memory(self, memory_id: str, target_tier: MemoryTier) -> bool:
        """Manually demote a memory to a lower tier."""
        return await self.tier_manager.demote(memory_id, target_tier)
    
    async def rebalance(self, agent_id: str) -> Dict[str, int]:
        """Rebalance memories across tiers."""
        return await self.tier_manager.rebalance_tiers(agent_id)
    
    async def cleanup(self, agent_id: str) -> int:
        """Clean up expired memories."""
        return await self.tier_manager.cleanup_expired(agent_id)
    
    async def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive tier statistics."""
        base = await self.backend.get_stats(agent_id)
        tier_stats = await self.tier_manager.get_tier_stats(agent_id)
        
        return {
            **base,
            "tiers": tier_stats,
        }


async def create_hierarchical_memory(
    agent_id: str = "default_agent",
    settings=None,
    backend: Optional[MemoryBackend] = None
) -> HierarchicalMemory:
    """Factory function to create a hierarchical memory system."""
    if backend is None:
        backend = LocalMemoryBackend(settings)
    
    memory = HierarchicalMemory(backend, settings)
    await memory.initialize()
    return memory