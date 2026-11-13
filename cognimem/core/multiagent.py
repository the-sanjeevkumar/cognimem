"""
CogniMem v2.0 - Multi-Agent Shared Memory
Team memory spaces with access control and conflict resolution.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

from pydantic import BaseModel, Field

from cognimem.config import get_settings
from cognimem.backends.base import MemoryBackend, MemoryRecord, MemoryQuery, SearchResult


class PermissionLevel(str, Enum):
    """Permission levels for shared memory."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving concurrent edits."""
    LAST_WRITE_WINS = "last_write_wins"
    MERGE = "merge"
    MANUAL = "manual"
    VECTOR_CLOCK = "vector_clock"


class MemoryNamespace(str, Enum):
    """Memory namespace types."""
    PERSONAL = "personal"      # Private to agent
    TEAM = "team"              # Shared within team
    ORGANIZATION = "org"       # Shared across organization
    PUBLIC = "public"          # Publicly accessible


@dataclass
class AccessControlEntry:
    """Access control entry for a memory namespace."""
    principal_id: str  # agent_id or team_id
    principal_type: str  # "agent" or "team"
    permission: PermissionLevel
    granted_by: str
    granted_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass
class SharedMemorySpace:
    """A shared memory space."""
    space_id: str
    name: str
    namespace: MemoryNamespace
    owner_id: str
    members: Dict[str, PermissionLevel] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryVersion(BaseModel):
    """Version of a memory record."""
    version: int
    content: str
    metadata: Dict[str, Any]
    author_id: str
    timestamp: datetime
    parent_version: Optional[int] = None
    change_summary: str = ""


class MultiAgentMemoryManager:
    """Manages multi-agent shared memory with access control."""
    
    def __init__(self, backend: MemoryBackend, settings=None):
        self.backend = backend
        self.settings = settings or get_settings()
        
        # In-memory state (would be persisted in production)
        self.spaces: Dict[str, SharedMemorySpace] = {}
        self.access_control: Dict[str, List[AccessControlEntry]] = defaultdict(list)
        self.memory_versions: Dict[str, List[MemoryVersion]] = defaultdict(list)
        self.team_memberships: Dict[str, Set[str]] = defaultdict(set)  # team_id -> agent_ids
        self.agent_teams: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> team_ids
        
        # Conflict resolution
        self.conflict_strategy = ConflictResolutionStrategy.LAST_WRITE_WINS
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    
    # Space management
    async def create_space(
        self,
        name: str,
        owner_id: str,
        namespace: MemoryNamespace = MemoryNamespace.TEAM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SharedMemorySpace:
        """Create a new shared memory space."""
        space_id = f"{namespace.value}_{uuid.uuid4().hex[:8]}"
        
        space = SharedMemorySpace(
            space_id=space_id,
            name=name,
            namespace=namespace,
            owner_id=owner_id,
            metadata=metadata or {}
        )
        
        # Owner gets admin access
        space.members[owner_id] = PermissionLevel.ADMIN
        
        self.spaces[space_id] = space
        
        # Record in ACL
        self.access_control[space_id].append(AccessControlEntry(
            principal_id=owner_id,
            principal_type="agent",
            permission=PermissionLevel.ADMIN,
            granted_by=owner_id
        ))
        
        return space
    
    async def get_space(self, space_id: str) -> Optional[SharedMemorySpace]:
        """Get a shared memory space."""
        return self.spaces.get(space_id)
    
    async def list_spaces(self, agent_id: str) -> List[SharedMemorySpace]:
        """List spaces accessible to an agent."""
        accessible = []
        
        # Direct membership
        for space in self.spaces.values():
            if agent_id in space.members:
                accessible.append(space)
            elif space.namespace == MemoryNamespace.PUBLIC:
                accessible.append(space)
        
        # Team membership
        for team_id in self.agent_teams.get(agent_id, set()):
            team_space_id = f"team_{team_id}"
            if team_space_id in self.spaces:
                space = self.spaces[team_space_id]
                if space not in accessible:
                    accessible.append(space)
        
        return accessible
    
    # Membership management
    async def add_member(
        self,
        space_id: str,
        agent_id: str,
        permission: PermissionLevel,
        granted_by: str
    ) -> bool:
        """Add a member to a shared space."""
        space = self.spaces.get(space_id)
        if not space:
            return False
        
        # Check grantor has admin permission
        grantor_perm = space.members.get(granted_by)
        if grantor_perm != PermissionLevel.ADMIN:
            return False
        
        space.members[agent_id] = permission
        
        self.access_control[space_id].append(AccessControlEntry(
            principal_id=agent_id,
            principal_type="agent",
            permission=permission,
            granted_by=granted_by
        ))
        
        return True
    
    async def remove_member(self, space_id: str, agent_id: str, removed_by: str) -> bool:
        """Remove a member from a shared space."""
        space = self.spaces.get(space_id)
        if not space:
            return False
        
        # Check remover has admin permission
        remover_perm = space.members.get(removed_by)
        if remover_perm != PermissionLevel.ADMIN:
            return False
        
        if agent_id == space.owner_id:
            return False  # Cannot remove owner
        
        space.members.pop(agent_id, None)
        
        # Add revocation entry
        self.access_control[space_id].append(AccessControlEntry(
            principal_id=agent_id,
            principal_type="agent",
            permission=PermissionLevel.NONE,
            granted_by=removed_by
        ))
        
        return True
    
    async def update_permission(
        self,
        space_id: str,
        agent_id: str,
        permission: PermissionLevel,
        updated_by: str
    ) -> bool:
        """Update a member's permission."""
        space = self.spaces.get(space_id)
        if not space:
            return False
        
        updater_perm = space.members.get(updated_by)
        if updater_perm != PermissionLevel.ADMIN:
            return False
        
        space.members[agent_id] = permission
        
        self.access_control[space_id].append(AccessControlEntry(
            principal_id=agent_id,
            principal_type="agent",
            permission=permission,
            granted_by=updated_by
        ))
        
        return True
    
    # Team management
    async def create_team(self, team_id: str, owner_id: str, name: str) -> bool:
        """Create a team."""
        if team_id in self.team_memberships:
            return False
        
        self.team_memberships[team_id] = {owner_id}
        self.agent_teams[owner_id].add(team_id)
        
        # Create team space
        await self.create_space(
            name=f"Team: {name}",
            owner_id=owner_id,
            namespace=MemoryNamespace.TEAM,
            metadata={"team_id": team_id, "team_name": name}
        )
        
        return True
    
    async def add_to_team(self, team_id: str, agent_id: str) -> bool:
        """Add agent to team."""
        if team_id not in self.team_memberships:
            return False
        
        self.team_memberships[team_id].add(agent_id)
        self.agent_teams[agent_id].add(team_id)
        return True
    
    async def remove_from_team(self, team_id: str, agent_id: str) -> bool:
        """Remove agent from team."""
        if team_id not in self.team_memberships:
            return False
        
        self.team_memberships[team_id].discard(agent_id)
        self.agent_teams[agent_id].discard(team_id)
        return True
    
    # Memory operations with access control
    async def _check_permission(
        self,
        space_id: str,
        agent_id: str,
        required: PermissionLevel
    ) -> bool:
        """Check if agent has required permission for space."""
        space = self.spaces.get(space_id)
        if not space:
            return False
        
        # Check direct permission
        perm = space.members.get(agent_id)
        if perm and self._perm_level(perm) >= self._perm_level(required):
            return True
        
        # Check team permission
        for team_id in self.agent_teams.get(agent_id, set()):
            team_space_id = f"team_{team_id}"
            if team_space_id == space_id:
                # Team members get at least read access
                return self._perm_level(PermissionLevel.READ) >= self._perm_level(required)
        
        # Public spaces
        if space.namespace == MemoryNamespace.PUBLIC:
            return self._perm_level(PermissionLevel.READ) >= self._perm_level(required)
        
        return False
    
    def _perm_level(self, perm: PermissionLevel) -> int:
        levels = {
            PermissionLevel.NONE: 0,
            PermissionLevel.READ: 1,
            PermissionLevel.WRITE: 2,
            PermissionLevel.ADMIN: 3,
        }
        return levels.get(perm, 0)
    
    async def remember_shared(
        self,
        space_id: str,
        agent_id: str,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Store a memory in a shared space."""
        # Check write permission
        if not await self._check_permission(space_id, agent_id, PermissionLevel.WRITE):
            return None
        
        tags = tags or []
        metadata = metadata or {}
        metadata["space_id"] = space_id
        tags.append(f"space:{space_id}")
        
        record = MemoryRecord(
            agent_id=agent_id,
            content=content,
            memory_type="semantic",
            tier="long_term",
            importance_score=importance,
            tags=tags,
            metadata=metadata,
        )
        
        # Use lock for conflict prevention
        async with self._locks[record.id]:
            memory_id = await self.backend.add(record)
            
            # Create initial version
            self.memory_versions[memory_id].append(MemoryVersion(
                version=1,
                content=content,
                metadata=metadata,
                author_id=agent_id,
                timestamp=datetime.utcnow(),
                change_summary="Initial creation"
            ))
            
            return memory_id
    
    async def recall_shared(
        self,
        space_id: str,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Recall memories from a shared space."""
        # Check read permission
        if not await self._check_permission(space_id, agent_id, PermissionLevel.READ):
            return []
        
        memory_query = MemoryQuery(
            query=query,
            agent_id=agent_id,
            top_k=top_k,
            filters={"space_id": space_id}
        )
        
        return await self.backend.search(memory_query)
    
    async def update_shared(
        self,
        space_id: str,
        agent_id: str,
        memory_id: str,
        new_content: str,
        change_summary: str = ""
    ) -> bool:
        """Update a shared memory with conflict resolution."""
        # Check write permission
        if not await self._check_permission(space_id, agent_id, PermissionLevel.WRITE):
            return False
        
        async with self._locks[memory_id]:
            # Get current record
            current = await self.backend.get(memory_id)
            if not current:
                return False
            
            # Check space ownership
            if current.metadata.get("space_id") != space_id:
                return False
            
            # Apply conflict resolution
            if self.conflict_strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
                # Simple overwrite
                pass
            elif self.conflict_strategy == ConflictResolutionStrategy.MERGE:
                # TODO: Implement merge logic
                pass
            
            # Create new version
            versions = self.memory_versions[memory_id]
            new_version = len(versions) + 1
            
            versions.append(MemoryVersion(
                version=new_version,
                content=new_content,
                metadata=current.metadata,
                author_id=agent_id,
                timestamp=datetime.utcnow(),
                parent_version=new_version - 1,
                change_summary=change_summary
            ))
            
            # Update record
            current.content = new_content
            current.version = new_version
            current.metadata["last_modified_by"] = agent_id
            current.metadata["last_modified_at"] = datetime.utcnow().isoformat()
            
            await self.backend.update(current)
            return True
    
    async def get_memory_history(self, memory_id: str, agent_id: str) -> List[MemoryVersion]:
        """Get version history of a memory."""
        # Check if agent has access to this memory
        record = await self.backend.get(memory_id)
        if not record:
            return []
        
        space_id = record.metadata.get("space_id")
        if space_id and not await self._check_permission(space_id, agent_id, PermissionLevel.READ):
            return []
        
        return self.memory_versions.get(memory_id, [])
    
    async def resolve_conflict(
        self,
        memory_id: str,
        conflicting_versions: List[MemoryVersion],
        resolution: str,
        resolved_by: str
    ) -> Optional[MemoryVersion]:
        """Manually resolve a conflict."""
        # This would be called when conflict_strategy is MANUAL
        # For now, just pick the resolution
        for v in conflicting_versions:
            if v.change_summary == resolution:
                return v
        return None
    
    # Audit and monitoring
    async def get_audit_log(
        self,
        space_id: str,
        agent_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit log for a space."""
        if not await self._check_permission(space_id, agent_id, PermissionLevel.ADMIN):
            return []
        
        entries = self.access_control.get(space_id, [])
        return [
            {
                "principal_id": e.principal_id,
                "principal_type": e.principal_type,
                "permission": e.permission.value,
                "granted_by": e.granted_by,
                "granted_at": e.granted_at.isoformat(),
                "expires_at": e.expires_at.isoformat() if e.expires_at else None,
            }
            for e in entries[-limit:]
        ]
    
    async def get_space_stats(self, space_id: str, agent_id: str) -> Dict[str, Any]:
        """Get statistics for a shared space."""
        if not await self._check_permission(space_id, agent_id, PermissionLevel.READ):
            return {}
        
        space = self.spaces.get(space_id)
        if not space:
            return {}
        
        # Count memories in this space
        all_memories = await self.backend.get_recent(space_id, limit=10000)
        space_memories = [m for m in all_memories if m.metadata.get("space_id") == space_id]
        
        return {
            "space_id": space_id,
            "name": space.name,
            "namespace": space.namespace.value,
            "member_count": len(space.members),
            "memory_count": len(space_memories),
            "created_at": space.created_at.isoformat(),
        }


async def create_multiagent_manager(backend: MemoryBackend, settings=None) -> MultiAgentMemoryManager:
    """Factory function to create a multi-agent memory manager."""
    return MultiAgentMemoryManager(backend, settings)