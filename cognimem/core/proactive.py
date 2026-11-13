"""
CogniMem v2.0 - Proactive Memory & Anticipatory Retrieval
Background memory maintenance and context-aware pre-fetching.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from pydantic import BaseModel, Field

from cognimem.config import get_settings
from cognimem.backends.base import MemoryBackend, MemoryRecord, MemoryQuery, SearchResult
from cognimem.core.query_planner import QueryPlanner, create_query_planner


class TriggerType(str, Enum):
    """Types of proactive triggers."""
    CONTEXT_CHANGE = "context_change"       # Agent context shifted
    TIME_BASED = "time_based"               # Scheduled retrieval
    PATTERN_MATCH = "pattern_match"         # Recognized recurring pattern
    GOAL_ALIGNMENT = "goal_alignment"       # Memory relevant to current goal
    ENTITY_MENTION = "entity_mention"       # Key entity mentioned
    DECISION_PENDING = "decision_pending"   # Agent needs to decide
    ERROR_RECOVERY = "error_recovery"       # Previous error context


@dataclass
class ProactiveTrigger:
    """A proactive retrieval trigger."""
    trigger_type: TriggerType
    agent_id: str
    context: Dict[str, Any]
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PrefetchResult:
    """Result of a prefetch operation."""
    trigger: ProactiveTrigger
    memories: List[SearchResult]
    relevance_scores: List[float]
    execution_time_ms: float
    success: bool
    error: Optional[str] = None


class PrefetchStrategy(ABC):
    """Abstract base for prefetch strategies."""
    
    @abstractmethod
    async def should_prefetch(self, trigger: ProactiveTrigger) -> bool:
        pass
    
    @abstractmethod
    async def generate_queries(self, trigger: ProactiveTrigger) -> List[str]:
        pass
    
    @abstractmethod
    async def rank_results(self, trigger: ProactiveTrigger, results: List[SearchResult]) -> List[SearchResult]:
        pass


class ContextAwarePrefetchStrategy(PrefetchStrategy):
    """Prefetch based on agent context changes."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._recent_triggers: Dict[str, List[ProactiveTrigger]] = defaultdict(list)
        self._max_history = 100
    
    async def should_prefetch(self, trigger: ProactiveTrigger) -> bool:
        # Don't prefetch too frequently
        recent = self._recent_triggers.get(trigger.agent_id, [])
        if len(recent) > 10:
            last_trigger = recent[-1]
            if (trigger.created_at - last_trigger.created_at).total_seconds() < 30:
                return False
        return True
    
    async def generate_queries(self, trigger: ProactiveTrigger) -> List[str]:
        context = trigger.context
        queries = []
        
        # Goal-based queries
        if "current_goal" in context:
            queries.append(f"Information relevant to: {context['current_goal']}")
        
        # Entity-based queries
        if "recent_entities" in context:
            for entity in context["recent_entities"][-3:]:
                queries.append(f"Details about {entity}")
        
        # Topic-based queries
        if "current_topic" in context:
            queries.append(f"Background on {context['current_topic']}")
        
        # Decision support
        if "pending_decision" in context:
            queries.append(f"Past decisions similar to: {context['pending_decision']}")
        
        return queries[:5]
    
    async def rank_results(self, trigger: ProactiveTrigger, results: List[SearchResult]) -> List[SearchResult]:
        # Boost results matching current context
        context = trigger.context
        boosted = []
        
        for result in results:
            score_boost = 0.0
            
            # Boost if matches current goal
            if "current_goal" in context:
                goal = context["current_goal"].lower()
                if goal in result.record.content.lower():
                    score_boost += 0.2
            
            # Boost if matches recent entities
            if "recent_entities" in context:
                for entity in context["recent_entities"]:
                    if entity.lower() in result.record.content.lower():
                        score_boost += 0.1
            
            # Boost if matches current topic
            if "current_topic" in context:
                topic = context["current_topic"].lower()
                if topic in result.record.content.lower():
                    score_boost += 0.15
            
            # Apply boost
            boosted_result = SearchResult(
                record=result.record,
                similarity_score=result.similarity_score,
                importance_score=result.importance_score,
                recency_score=result.recency_score,
                final_score=min(1.0, result.final_score + score_boost),
                matched_fields=result.matched_fields + (["context_boost"] if score_boost > 0 else [])
            )
            boosted.append(boosted_result)
        
        boosted.sort(key=lambda x: x.final_score, reverse=True)
        return boosted
    
    def record_trigger(self, trigger: ProactiveTrigger) -> None:
        """Record a trigger for rate limiting."""
        self._recent_triggers[trigger.agent_id].append(trigger)
        if len(self._recent_triggers[trigger.agent_id]) > self._max_history:
            self._recent_triggers[trigger.agent_id] = self._recent_triggers[trigger.agent_id][-self._max_history:]


class TemporalPrefetchStrategy(PrefetchStrategy):
    """Time-based prefetching (e.g., daily summaries, weekly reviews)."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._schedules: Dict[str, Dict[str, Any]] = {}
    
    async def should_prefetch(self, trigger: ProactiveTrigger) -> bool:
        return trigger.trigger_type == TriggerType.TIME_BASED
    
    async def generate_queries(self, trigger: ProactiveTrigger) -> List[str]:
        schedule = trigger.metadata.get("schedule", "daily")
        queries = []
        
        if schedule == "daily":
            queries.extend([
                "Summary of yesterday's activities",
                "Key decisions made yesterday",
                "Commitments for today"
            ])
        elif schedule == "weekly":
            queries.extend([
                "Weekly summary of accomplishments",
                "Patterns and trends this week",
                "Outstanding commitments"
            ])
        elif schedule == "monthly":
            queries.extend([
                "Monthly progress review",
                "Long-term goal alignment",
                "Strategic decisions made"
            ])
        
        return queries
    
    async def rank_results(self, trigger: ProactiveTrigger, results: List[SearchResult]) -> List[SearchResult]:
        # For temporal prefetch, prioritize recency
        return sorted(results, key=lambda x: x.recency_score, reverse=True)


class PatternBasedPrefetchStrategy(PrefetchStrategy):
    """Prefetch based on recognized usage patterns."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._query_patterns: Dict[str, List[str]] = defaultdict(list)
        self._co_occurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    async def should_prefetch(self, trigger: ProactiveTrigger) -> bool:
        return trigger.trigger_type == TriggerType.PATTERN_MATCH
    
    async def generate_queries(self, trigger: ProactiveTrigger) -> List[str]:
        context = trigger.context
        last_query = context.get("last_query", "")
        
        if not last_query:
            return []
        
        # Find co-occurring queries
        related = self._co_occurrence.get(last_query, {})
        if not related:
            return []
        
        # Get top co-occurring queries
        sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
        return [q for q, _ in sorted_related[:3]]
    
    async def rank_results(self, trigger: ProactiveTrigger, results: List[SearchResult]) -> List[SearchResult]:
        return results  # No special ranking
    
    def record_query_pair(self, agent_id: str, query1: str, query2: str) -> None:
        """Record that two queries occurred in sequence."""
        key = f"{agent_id}:{query1}"
        self._co_occurrence[key][query2] += 1
        
        # Keep only top 20 per query
        if len(self._co_occurrence[key]) > 20:
            sorted_items = sorted(self._co_occurrence[key].items(), key=lambda x: x[1], reverse=True)
            self._co_occurrence[key] = dict(sorted_items[:20])


class ProactiveMemoryManager:
    """Manages proactive memory retrieval."""
    
    def __init__(
        self, 
        backend: MemoryBackend, 
        query_planner: QueryPlanner,
        settings=None
    ):
        self.backend = backend
        self.query_planner = query_planner
        self.settings = settings or get_settings()
        
        self.strategies: List[PrefetchStrategy] = [
            ContextAwarePrefetchStrategy(settings),
            TemporalPrefetchStrategy(settings),
            PatternBasedPrefetchStrategy(settings),
        ]
        
        self._prefetch_cache: Dict[str, List[SearchResult]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._callbacks: List[Callable[[PrefetchResult], Any]] = []
        self._running = False
        self._background_task: Optional[asyncio.Task] = None
    
    def register_strategy(self, strategy: PrefetchStrategy) -> None:
        """Register a custom prefetch strategy."""
        self.strategies.append(strategy)
    
    def register_callback(self, callback: Callable[[PrefetchResult], Any]) -> None:
        """Register callback for prefetch results."""
        self._callbacks.append(callback)
    
    async def start_background_prefetch(self, interval_seconds: int = 60) -> None:
        """Start background prefetch loop."""
        if self._running:
            return
        
        self._running = True
        self._background_task = asyncio.create_task(self._prefetch_loop(interval_seconds))
    
    async def stop_background_prefetch(self) -> None:
        """Stop background prefetch loop."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
    
    async def _prefetch_loop(self, interval: int) -> None:
        """Background loop for time-based prefetching."""
        while self._running:
            try:
                # Check for scheduled prefetches
                await self._check_scheduled_prefetches()
            except Exception:
                pass
            
            await asyncio.sleep(interval)
    
    async def _check_scheduled_prefetches(self) -> None:
        """Check and execute scheduled prefetches."""
        # This would integrate with a scheduler
        # For now, just a placeholder
        pass
    
    async def trigger_prefetch(
        self,
        trigger: ProactiveTrigger,
        max_results: int = 5
    ) -> PrefetchResult:
        """Execute prefetch for a trigger."""
        start_time = datetime.utcnow()
        
        # Check cache first
        cache_key = f"{trigger.agent_id}:{trigger.trigger_type.value}:{hash(str(trigger.context))}"
        if cache_key in self._prefetch_cache:
            cached = self._prefetch_cache[cache_key]
            return PrefetchResult(
                trigger=trigger,
                memories=cached,
                relevance_scores=[r.final_score for r in cached],
                execution_time_ms=0.0,
                success=True
            )
        
        # Find applicable strategies
        applicable = [s for s in self.strategies if await s.should_prefetch(trigger)]
        
        if not applicable:
            return PrefetchResult(
                trigger=trigger,
                memories=[],
                relevance_scores=[],
                execution_time_ms=0.0,
                success=False,
                error="No applicable strategy"
            )
        
        all_results = []
        for strategy in applicable:
            try:
                queries = await strategy.generate_queries(trigger)
                
                for query in queries:
                    # Check cache for this query
                    qcache_key = f"{trigger.agent_id}:{query}"
                    if qcache_key in self._prefetch_cache:
                        all_results.extend(self._prefetch_cache[qcache_key])
                        continue
                    
                    # Execute query
                    memory_query = MemoryQuery(
                        query=query,
                        agent_id=trigger.agent_id,
                        top_k=max_results
                    )
                    results = await self.backend.search(memory_query)
                    
                    # Rank with strategy
                    ranked = await strategy.rank_results(trigger, results)
                    all_results.extend(ranked)
                    
                    # Cache
                    self._prefetch_cache[qcache_key] = ranked[:max_results]
            
            except Exception as e:
                continue
        
        # Deduplicate and sort
        seen = set()
        unique_results = []
        for r in all_results:
            if r.record.id not in seen:
                seen.add(r.record.id)
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x.final_score, reverse=True)
        final_results = unique_results[:max_results]
        
        # Cache final results
        self._prefetch_cache[cache_key] = final_results
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        result = PrefetchResult(
            trigger=trigger,
            memories=final_results,
            relevance_scores=[r.final_score for r in final_results],
            execution_time_ms=execution_time,
            success=len(final_results) > 0
        )
        
        # Record trigger for rate limiting
        for strategy in self.strategies:
            if isinstance(strategy, ContextAwarePrefetchStrategy):
                strategy.record_trigger(trigger)
        
        # Fire callbacks
        for callback in self._callbacks:
            try:
                await callback(result)
            except Exception:
                pass
        
        return result
    
    async def on_context_change(
        self,
        agent_id: str,
        context: Dict[str, Any],
        max_results: int = 3
    ) -> PrefetchResult:
        """Handle context change event."""
        trigger = ProactiveTrigger(
            trigger_type=TriggerType.CONTEXT_CHANGE,
            agent_id=agent_id,
            context=context,
            priority=2
        )
        return await self.trigger_prefetch(trigger, max_results)
    
    async def on_entity_mention(
        self,
        agent_id: str,
        entity: str,
        context: Dict[str, Any],
        max_results: int = 3
    ) -> PrefetchResult:
        """Handle entity mention event."""
        context["recent_entities"] = context.get("recent_entities", []) + [entity]
        context["last_entity"] = entity
        
        trigger = ProactiveTrigger(
            trigger_type=TriggerType.ENTITY_MENTION,
            agent_id=agent_id,
            context=context,
            priority=2
        )
        return await self.trigger_prefetch(trigger, max_results)
    
    async def on_decision_pending(
        self,
        agent_id: str,
        decision: str,
        context: Dict[str, Any],
        max_results: int = 5
    ) -> PrefetchResult:
        """Handle pending decision event."""
        context["pending_decision"] = decision
        
        trigger = ProactiveTrigger(
            trigger_type=TriggerType.DECISION_PENDING,
            agent_id=agent_id,
            context=context,
            priority=3
        )
        return await self.trigger_prefetch(trigger, max_results)
    
    def get_cached_prefetch(self, agent_id: str, query: str) -> Optional[List[SearchResult]]:
        """Get cached prefetch results."""
        cache_key = f"{agent_id}:{query}"
        return self._prefetch_cache.get(cache_key)
    
    def clear_cache(self, agent_id: Optional[str] = None) -> None:
        """Clear prefetch cache."""
        if agent_id:
            keys_to_remove = [k for k in self._prefetch_cache if k.startswith(f"{agent_id}:")]
            for k in keys_to_remove:
                del self._prefetch_cache[k]
        else:
            self._prefetch_cache.clear()


async def create_proactive_manager(
    backend: MemoryBackend,
    settings=None
) -> ProactiveMemoryManager:
    """Factory function to create a proactive memory manager."""
    query_planner = await create_query_planner(settings)
    return ProactiveMemoryManager(backend, query_planner, settings)