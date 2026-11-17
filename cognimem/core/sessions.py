"""
CogniMem v2.0 - Session Management & Cross-Session Continuity
Persistent sessions with automatic summarization and bridging.
"""

import asyncio
import uuid
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from collections import defaultdict

from pydantic import BaseModel, Field

from cognimem.time_utils import utcnow
from cognimem.config import get_settings
from cognimem.backends.base import MemoryBackend, MemoryRecord, MemoryQuery, SearchResult
from cognimem.core.compression import MemoryCompressor, create_memory_compressor


class SessionStatus(str, Enum):
    """Session lifecycle status."""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ARCHIVED = "archived"


def _as_utc(dt: datetime) -> datetime:
    """Treat naive timestamps as UTC so comparisons never raise TypeError."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


@dataclass
class Session:
    """A conversation/memory session."""
    session_id: str
    agent_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    ended_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_count: int = 0
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    parent_session_id: Optional[str] = None
    child_session_ids: List[str] = field(default_factory=list)


@dataclass
class SessionBridge:
    """Bridge between related sessions."""
    bridge_id: str
    from_session_id: str
    to_session_id: str
    bridge_type: str  # "continuation", "reference", "context_transfer"
    created_at: datetime = field(default_factory=utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Manages memory sessions with cross-session continuity."""
    
    def __init__(self, backend: MemoryBackend, compressor: MemoryCompressor, settings=None):
        self.backend = backend
        self.compressor = compressor
        self.settings = settings or get_settings()
        self._initialized = False
        
        # In-memory session state
        self.sessions: Dict[str, Session] = {}
        self.session_bridges: Dict[str, SessionBridge] = {}
        self.agent_sessions: Dict[str, List[str]] = defaultdict(list)  # agent_id -> session_ids
    
    async def initialize(self) -> None:
        """Initialize the session manager."""
        if self._initialized:
            return
        await self.backend.initialize()
        self._initialized = True
        
        # Configuration
        self.max_session_duration_hours = 24
        self.auto_summarize_on_end = True
        self.bridge_context_window = 5  # Number of memories to bridge
    
    async def create_session(
        self,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_session_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Session:
        """Create a new session."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        session = Session(
            session_id=session_id,
            agent_id=agent_id,
            metadata=metadata or {},
            tags=tags or [],
            parent_session_id=parent_session_id
        )
        
        self.sessions[session_id] = session
        self.agent_sessions[agent_id].append(session_id)
        
        # Link to parent
        if parent_session_id and parent_session_id in self.sessions:
            parent = self.sessions[parent_session_id]
            parent.child_session_ids.append(session_id)
            
            # Create bridge
            bridge = SessionBridge(
                bridge_id=f"bridge_{uuid.uuid4().hex[:8]}",
                from_session_id=parent_session_id,
                to_session_id=session_id,
                bridge_type="continuation",
                metadata={"created_by": "session_manager"}
            )
            self.session_bridges[bridge.bridge_id] = bridge
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    async def get_active_session(self, agent_id: str) -> Optional[Session]:
        """Get the current active session for an agent."""
        for session_id in reversed(self.agent_sessions.get(agent_id, [])):
            session = self.sessions.get(session_id)
            if session and session.status == SessionStatus.ACTIVE:
                return session
        return None
    
    async def end_session(self, session_id: str, create_summary: bool = True) -> bool:
        """End a session and optionally create a summary."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session.status = SessionStatus.ENDED
        session.ended_at = utcnow()
        session.updated_at = utcnow()
        
        if create_summary and self.auto_summarize_on_end:
            await self._create_session_summary(session)
        
        return True
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a session."""
        session = self.sessions.get(session_id)
        if not session or session.status != SessionStatus.ACTIVE:
            return False
        
        session.status = SessionStatus.PAUSED
        session.updated_at = utcnow()
        return True
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        session = self.sessions.get(session_id)
        if not session or session.status != SessionStatus.PAUSED:
            return False
        
        session.status = SessionStatus.ACTIVE
        session.updated_at = utcnow()
        return True
    
    async def archive_session(self, session_id: str) -> bool:
        """Archive an ended session."""
        session = self.sessions.get(session_id)
        if not session or session.status != SessionStatus.ENDED:
            return False
        
        session.status = SessionStatus.ARCHIVED
        session.updated_at = utcnow()
        return True
    
    async def _create_session_summary(self, session: Session) -> None:
        """Create a summary of the session's memories."""
        # Get all memories for this session
        memories = await self.backend.get_recent(
            session.agent_id,
            limit=1000
        )
        
        # Filter to session time range
        session_memories = [
            m for m in memories
            if session.created_at <= m.timestamp <= (session.ended_at or utcnow())
        ]
        
        if not session_memories:
            session.summary = "No memories recorded in this session."
            return
        
        # Create hierarchical summary
        summaries = await self.compressor.create_hierarchical_summaries(session.agent_id, limit=len(session_memories))
        
        # Combine into session summary
        summary_parts = []
        if "session" in summaries:
            summary_parts.append(f"Session Summary: {summaries['session'].compressed_content}")
        elif "turn" in summaries:
            summary_parts.append(f"Key Points: {summaries['turn'].compressed_content}")
        
        if "day" in summaries:
            summary_parts.append(f"Daily Context: {summaries['day'].compressed_content}")
        
        session.summary = "\n\n".join(summary_parts)
        session.memory_count = len(session_memories)
    
    async def get_session_memories(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[MemoryRecord]:
        """Get all memories associated with a session."""
        session = self.sessions.get(session_id)
        if not session:
            return []
        
        memories = await self.backend.get_recent(
            session.agent_id,
            limit=limit * 2
        )
        
        # Filter by session time range
        end_time = session.ended_at or utcnow()
        session_memories = [
            m for m in memories
            if session.created_at <= m.timestamp <= end_time
        ]
        
        session_memories.sort(key=lambda x: x.timestamp)
        return session_memories[:limit]
    
    async def create_context_bridge(
        self,
        from_session_id: str,
        to_session_id: str,
        bridge_type: str = "context_transfer",
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionBridge:
        """Create a context bridge between sessions."""
        bridge = SessionBridge(
            bridge_id=f"bridge_{uuid.uuid4().hex[:8]}",
            from_session_id=from_session_id,
            to_session_id=to_session_id,
            bridge_type=bridge_type,
            metadata=metadata or {}
        )
        
        self.session_bridges[bridge.bridge_id] = bridge
        return bridge
    
    async def get_bridged_context(
        self,
        session_id: str,
        limit: int = 5
    ) -> List[MemoryRecord]:
        """Get context from bridged sessions."""
        # Find incoming bridges
        incoming_bridges = [
            b for b in self.session_bridges.values()
            if b.to_session_id == session_id
        ]
        
        context_memories = []
        for bridge in incoming_bridges:
            from_session = self.sessions.get(bridge.from_session_id)
            if not from_session:
                continue
            
            # Get recent memories from source session
            memories = await self.get_session_memories(bridge.from_session_id, limit=limit)
            context_memories.extend(memories)
        
        # Sort by timestamp, most recent first
        context_memories.sort(key=lambda x: x.timestamp, reverse=True)
        return context_memories[:limit]
    
    async def list_agent_sessions(
        self,
        agent_id: str,
        status: Optional[SessionStatus] = None,
        limit: int = 50
    ) -> List[Session]:
        """List all sessions for an agent."""
        session_ids = self.agent_sessions.get(agent_id, [])
        sessions = []
        
        for sid in reversed(session_ids):
            session = self.sessions.get(sid)
            if session:
                if status is None or session.status == status:
                    sessions.append(session)
                if len(sessions) >= limit:
                    break
        
        return sessions
    
    async def find_sessions_by_tag(
        self,
        agent_id: str,
        tag: str
    ) -> List[Session]:
        """Find sessions by tag."""
        sessions = await self.list_agent_sessions(agent_id)
        return [s for s in sessions if tag in s.tags]
    
    async def cleanup_old_sessions(
        self,
        agent_id: str,
        max_age_days: int = 30,
        archive: bool = True
    ) -> int:
        """Clean up old inactive sessions."""
        cutoff = utcnow() - timedelta(days=max_age_days)
        cleaned = 0
        
        for session_id in self.agent_sessions.get(agent_id, []):
            session = self.sessions.get(session_id)
            if not session:
                continue
            
            if session.updated_at < cutoff and session.status in [SessionStatus.ENDED, SessionStatus.PAUSED]:
                if archive:
                    await self.archive_session(session_id)
                else:
                    session.status = SessionStatus.ARCHIVED
                cleaned += 1
        
        return cleaned
    
    async def get_session_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get session statistics for an agent."""
        sessions = await self.list_agent_sessions(agent_id, limit=1000)
        
        total = len(sessions)
        active = sum(1 for s in sessions if s.status == SessionStatus.ACTIVE)
        ended = sum(1 for s in sessions if s.status == SessionStatus.ENDED)
        archived = sum(1 for s in sessions if s.status == SessionStatus.ARCHIVED)
        
        total_memories = sum(s.memory_count for s in sessions)
        
        return {
            "total_sessions": total,
            "active": active,
            "ended": ended,
            "archived": archived,
            "total_memories": total_memories,
            "avg_memories_per_session": total_memories / total if total > 0 else 0,
        }


class CrossSessionRetriever:
    """Retrieves memories across session boundaries."""
    
    def __init__(self, session_manager: SessionManager, backend: MemoryBackend):
        self.session_manager = session_manager
        self.backend = backend
    
    async def recall_with_context(
        self,
        query: str,
        agent_id: str,
        current_session_id: Optional[str] = None,
        top_k: int = 5,
        include_bridged: bool = True,
        include_historical: bool = True
    ) -> List[SearchResult]:
        """Recall memories with cross-session context."""
        all_results = []
        
        # 1. Current session memories (highest priority)
        if current_session_id:
            session_memories = await self.session_manager.get_session_memories(current_session_id)
            session_results = await self._memories_to_results(session_memories, query)
            for r in session_results:
                r.record.metadata["source"] = "current_session"
            all_results.extend(session_results)
        
        # 2. Bridged context from related sessions
        if include_bridged and current_session_id:
            bridged = await self.session_manager.get_bridged_context(current_session_id, limit=top_k)
            bridged_results = await self._memories_to_results(bridged, query)
            for r in bridged_results:
                r.record.metadata["source"] = "bridged_session"
            all_results.extend(bridged_results)
        
        # 3. Historical memories (lower priority)
        if include_historical:
            historical_results = await self.backend.search(MemoryQuery(
                query=query,
                agent_id=agent_id,
                top_k=top_k
            ))
            for r in historical_results:
                r.record.metadata["source"] = "historical"
            all_results.extend(historical_results)
        
        # Deduplicate and re-rank
        return self._deduplicate_and_rank(all_results, top_k)
    
    async def _memories_to_results(
        self,
        memories: List[MemoryRecord],
        query: str
    ) -> List[SearchResult]:
        """Convert memories to search results with relevance scoring."""
        # Simple relevance: check if query terms appear in content
        query_terms = set(query.lower().split())
        
        results = []
        for mem in memories:
            content_terms = set(mem.content.lower().split())
            overlap = len(query_terms & content_terms)
            relevance = overlap / len(query_terms) if query_terms else 0.5
            
            results.append(SearchResult(
                record=mem,
                similarity_score=relevance,
                importance_score=mem.importance_score,
                recency_score=mem.importance_score,  # Simplified
                final_score=relevance * 0.7 + mem.importance_score * 0.3,
                matched_fields=["content"] if overlap > 0 else []
            ))
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    def _deduplicate_and_rank(
        self,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Deduplicate by memory ID and rank."""
        seen = set()
        unique = []
        
        # Priority order for sources
        source_priority = {
            "current_session": 3,
            "bridged_session": 2,
            "historical": 1
        }
        
        for r in results:
            if r.record.id not in seen:
                seen.add(r.record.id)
                # Boost score by source priority
                source = r.record.metadata.get("source", "historical")
                boost = source_priority.get(source, 1) * 0.1
                r.final_score = min(1.0, r.final_score + boost)
                unique.append(r)
        
        unique.sort(key=lambda x: x.final_score, reverse=True)
        return unique[:top_k]


async def create_session_manager(
    backend: MemoryBackend,
    settings=None
) -> SessionManager:
    """Factory function to create a session manager."""
    compressor = await create_memory_compressor(backend, settings)
    return SessionManager(backend, compressor, settings)


async def create_cross_session_retriever(
    session_manager: SessionManager,
    backend: MemoryBackend
) -> CrossSessionRetriever:
    """Factory function to create a cross-session retriever."""
    return CrossSessionRetriever(session_manager, backend)