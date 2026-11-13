"""
CogniMem v2.0 - Test Suite
Comprehensive tests for the advanced cognitive memory system.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from cognimem.config import CogniMemSettings, EmbeddingConfig
from cognimem.backends.base import MemoryRecord, MemoryQuery, SearchResult
from cognimem.core.hierarchical import MemoryTier, TierManager, TierConfig, HierarchicalMemory
from cognimem.core.compression import CompressionLevel, CompressionResult
from cognimem.core.query_planner import QueryType, QueryIntent, SubQuery, QueryPlan
from cognimem.core.proactive import TriggerType, ProactiveTrigger, PrefetchResult
from cognimem.core.multiagent import PermissionLevel, MemoryNamespace, SharedMemorySpace
from cognimem.core.sessions import Session, SessionStatus, SessionBridge


# Fixtures
@pytest.fixture
def mock_settings():
    return CogniMemSettings(agent_id="test_agent")


@pytest.fixture
def mock_backend():
    backend = AsyncMock()
    backend.initialize = AsyncMock()
    backend.close = AsyncMock()
    backend.health_check = AsyncMock(return_value=True)
    backend.add = AsyncMock(return_value="mem_123")
    backend.add_batch = AsyncMock(return_value=["mem_1", "mem_2"])
    backend.get = AsyncMock(return_value=None)
    backend.update = AsyncMock(return_value=True)
    backend.delete = AsyncMock(return_value=True)
    backend.delete_by_agent = AsyncMock(return_value=0)
    backend.search = AsyncMock(return_value=[])
    backend.search_by_embedding = AsyncMock(return_value=[])
    backend.get_by_tags = AsyncMock(return_value=[])
    backend.get_recent = AsyncMock(return_value=[])
    backend.get_by_time_range = AsyncMock(return_value=[])
    backend.promote_to_tier = AsyncMock(return_value=True)
    backend.demote_to_tier = AsyncMock(return_value=True)
    backend.get_tier_stats = AsyncMock(return_value={})
    backend.get_stats = AsyncMock(return_value={"total_memories": 0})
    backend.get_importance_distribution = AsyncMock(return_value={})
    return backend


@pytest.fixture
def sample_memory():
    return MemoryRecord(
        id="mem_123",
        agent_id="test_agent",
        content="Test memory content",
        memory_type="episodic",
        tier="short_term",
        importance_score=0.7,
        tags=["test", "sample"],
        metadata={}
    )


# Config Tests
class TestConfig:
    def test_default_settings(self):
        settings = CogniMemSettings()
        assert settings.agent_id == "default_agent"
        assert settings.embedding.model == "all-MiniLM-L6-v2"
        assert settings.memory.working_memory_limit == 10
    
    def test_custom_settings(self):
        settings = CogniMemSettings(
            agent_id="custom_agent",
            embedding=EmbeddingConfig(model="custom-model")
        )
        assert settings.agent_id == "custom_agent"
        assert settings.embedding.model == "custom-model"
    
    def test_tier_configs(self):
        settings = CogniMemSettings()
        assert settings.retrieval.vector_weight == 0.6
        assert settings.consolidation.decay_constant_k == 24.0


# MemoryRecord Tests
class TestMemoryRecord:
    def test_default_values(self):
        record = MemoryRecord(agent_id="test", content="test")
        assert record.id is not None
        assert record.timestamp is not None
        assert record.importance_score == 1.0
        assert record.tags == []
        assert record.metadata == {}
        assert record.version == 1
        assert record.compressed is False
    
    def test_custom_values(self):
        record = MemoryRecord(
            agent_id="test",
            content="test",
            importance_score=0.5,
            tags=["tag1", "tag2"],
            metadata={"key": "value"}
        )
        assert record.importance_score == 0.5
        assert record.tags == ["tag1", "tag2"]
        assert record.metadata == {"key": "value"}


# Hierarchical Memory Tests
class TestHierarchicalMemory:
    @pytest.mark.asyncio
    async def test_tier_manager_auto_tier(self, mock_backend):
        tier_manager = TierManager(mock_backend)
        
        # High importance -> long_term
        record = MemoryRecord(agent_id="test", content="test", importance_score=0.8, memory_type="episodic")
        tier = await tier_manager.auto_tier_memory(record)
        assert tier == MemoryTier.LONG_TERM
        
        # Medium importance -> short_term
        record.importance_score = 0.3
        record.memory_type = "episodic"
        tier = await tier_manager.auto_tier_memory(record)
        assert tier == MemoryTier.SHORT_TERM
        
        # Low importance, recent -> immediate
        record.importance_score = 0.05
        record.memory_type = "episodic"
        record.timestamp = datetime.utcnow()
        tier = await tier_manager.auto_tier_memory(record)
        assert tier == MemoryTier.IMMEDIATE
    
    @pytest.mark.asyncio
    async def test_tier_stats(self, mock_backend):
        tier_manager = TierManager(mock_backend)
        mock_backend.get_tier_stats.return_value = {
            "immediate": 5,
            "short_term": 50,
            "long_term": 500,
            "archival": 5000
        }
        
        stats = await tier_manager.get_tier_stats("test_agent")
        assert "immediate" in stats
        assert "utilization" in stats["immediate"]


# Compression Tests
class TestCompression:
    def test_compression_levels(self):
        assert CompressionLevel.NONE.value == "none"
        assert CompressionLevel.LIGHT.value == "light"
        assert CompressionLevel.MEDIUM.value == "medium"
        assert CompressionLevel.AGGRESSIVE.value == "aggressive"
    
    def test_compression_result(self):
        result = CompressionResult(
            original_content="Original content",
            compressed_content="Compressed",
            compression_ratio=0.5,
            level=CompressionLevel.MEDIUM,
            preserved_entities=["entity1"],
            preserved_decisions=["decision1"],
            preserved_commitments=["commitment1"]
        )
        assert result.compression_ratio == 0.5
        assert result.level == CompressionLevel.MEDIUM


# Query Planner Tests
class TestQueryPlanner:
    def test_query_types(self):
        assert QueryType.SIMPLE.value == "simple"
        assert QueryType.COMPLEX.value == "complex"
        assert QueryType.TEMPORAL.value == "temporal"
        assert QueryType.CAUSAL.value == "causal"
    
    def test_query_intent(self):
        intent = QueryIntent(
            query_type=QueryType.TEMPORAL,
            entities=["Alice", "Bob"],
            time_references=["yesterday", "last week"],
            intent_keywords=["when", "timeline"],
            requires_reasoning=True,
            suggested_sub_queries=["When did Alice arrive?", "What happened last week?"],
            confidence=0.9
        )
        assert intent.query_type == QueryType.TEMPORAL
        assert len(intent.entities) == 2
    
    def test_sub_query(self):
        sq = SubQuery(
            id="sq_1",
            query="Information about Alice",
            query_type=QueryType.SIMPLE,
            priority=1,
            dependencies=[],
            expected_result_type="entity_info"
        )
        assert sq.id == "sq_1"
        assert sq.priority == 1
    
    def test_query_plan(self):
        plan = QueryPlan(
            original_query="What happened yesterday?",
            intent=QueryIntent(query_type=QueryType.TEMPORAL),
            sub_queries=[
                SubQuery(id="sq_1", query="Events yesterday", query_type=QueryType.TEMPORAL)
            ],
            fusion_strategy="rrf",
            estimated_steps=1,
            parallelizable=True
        )
        assert len(plan.sub_queries) == 1
        assert plan.fusion_strategy == "rrf"


# Proactive Memory Tests
class TestProactiveMemory:
    def test_trigger_types(self):
        assert TriggerType.CONTEXT_CHANGE.value == "context_change"
        assert TriggerType.ENTITY_MENTION.value == "entity_mention"
        assert TriggerType.DECISION_PENDING.value == "decision_pending"
    
    def test_proactive_trigger(self):
        trigger = ProactiveTrigger(
            trigger_type=TriggerType.CONTEXT_CHANGE,
            agent_id="test_agent",
            context={"current_goal": "Deploy app"},
            priority=2
        )
        assert trigger.trigger_type == TriggerType.CONTEXT_CHANGE
        assert trigger.agent_id == "test_agent"
        assert trigger.priority == 2
    
    def test_prefetch_result(self):
        result = PrefetchResult(
            trigger=ProactiveTrigger(
                trigger_type=TriggerType.TIME_BASED,
                agent_id="test",
                context={}
            ),
            memories=[],
            relevance_scores=[],
            execution_time_ms=10.5,
            success=True
        )
        assert result.success is True
        assert result.execution_time_ms == 10.5


# Multi-Agent Tests
class TestMultiAgent:
    def test_permission_levels(self):
        assert PermissionLevel.NONE.value == "none"
        assert PermissionLevel.READ.value == "read"
        assert PermissionLevel.WRITE.value == "write"
        assert PermissionLevel.ADMIN.value == "admin"
    
    def test_memory_namespaces(self):
        assert MemoryNamespace.PERSONAL.value == "personal"
        assert MemoryNamespace.TEAM.value == "team"
        assert MemoryNamespace.ORGANIZATION.value == "org"
        assert MemoryNamespace.PUBLIC.value == "public"
    
    def test_shared_memory_space(self):
        space = SharedMemorySpace(
            space_id="space_123",
            name="Test Space",
            namespace=MemoryNamespace.TEAM,
            owner_id="agent_1",
            members={"agent_1": PermissionLevel.ADMIN, "agent_2": PermissionLevel.WRITE}
        )
        assert space.space_id == "space_123"
        assert space.namespace == MemoryNamespace.TEAM
        assert space.members["agent_1"] == PermissionLevel.ADMIN


# Session Tests
class TestSessions:
    def test_session_status(self):
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.ENDED.value == "ended"
        assert SessionStatus.ARCHIVED.value == "archived"
    
    def test_session(self):
        session = Session(
            session_id="sess_123",
            agent_id="test_agent",
            status=SessionStatus.ACTIVE,
            tags=["meeting", "demo"]
        )
        assert session.session_id == "sess_123"
        assert session.status == SessionStatus.ACTIVE
        assert "meeting" in session.tags
    
    def test_session_bridge(self):
        bridge = SessionBridge(
            bridge_id="bridge_123",
            from_session_id="sess_1",
            to_session_id="sess_2",
            bridge_type="continuation"
        )
        assert bridge.from_session_id == "sess_1"
        assert bridge.to_session_id == "sess_2"
        assert bridge.bridge_type == "continuation"


# Integration Tests
class TestIntegration:
    @pytest.mark.asyncio
    async def test_hierarchical_memory_remember(self, mock_backend, mock_settings):
        mock_backend.add.return_value = "new_mem_id"
        
        hierarchical = HierarchicalMemory(mock_backend, mock_settings)
        await hierarchical.initialize()
        
        memory_id = await hierarchical.remember(
            content="Test content",
            agent_id="test_agent",
            importance=0.8,
            tags=["test"]
        )
        
        assert memory_id == "new_mem_id"
        mock_backend.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_hierarchical_memory_recall(self, mock_backend, mock_settings):
        from cognimem.backends.base import SearchResult
        
        mock_result = SearchResult(
            record=MemoryRecord(
                id="mem_1",
                agent_id="test_agent",
                content="Test result",
                memory_type="episodic",
                tier="short_term",
                importance_score=0.7,
                tags=["test"]
            ),
            similarity_score=0.8,
            importance_score=0.7,
            recency_score=0.6,
            final_score=0.72
        )
        mock_backend.search.return_value = [mock_result]
        
        hierarchical = HierarchicalMemory(mock_backend, mock_settings)
        await hierarchical.initialize()
        
        results = await hierarchical.recall(
            query="test query",
            agent_id="test_agent",
            top_k=5
        )
        
        assert len(results) == 1
        assert results[0].record.content == "Test result"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])