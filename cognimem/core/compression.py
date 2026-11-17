"""
CogniMem v2.0 - Memory Compression & Summarization
LLM-based conversation summarization with hierarchical compression.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from cognimem.time_utils import utcnow
from cognimem.config import get_settings
from cognimem.backends.base import MemoryBackend, MemoryRecord


class CompressionLevel(str, Enum):
    """Compression levels."""
    NONE = "none"
    LIGHT = "light"      # Remove redundancy, keep details
    MEDIUM = "medium"    # Summarize with key points
    AGGRESSIVE = "aggressive"  # Keep only decisions/entities/commitments


@dataclass
class CompressionResult:
    """Result of compression operation."""
    original_content: str
    compressed_content: str
    compression_ratio: float
    level: CompressionLevel
    preserved_entities: List[str] = field(default_factory=list)
    preserved_decisions: List[str] = field(default_factory=list)
    preserved_commitments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SummarizationStrategy(ABC):
    """Abstract base for summarization strategies."""
    
    @abstractmethod
    async def summarize(
        self, 
        content: str, 
        level: CompressionLevel,
        context: Optional[Dict[str, Any]] = None
    ) -> CompressionResult:
        pass


class LLMSummarizationStrategy(SummarizationStrategy):
    """LLM-based summarization using OpenAI-compatible API."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client = None
    
    async def _get_client(self):
        """Lazy initialize the LLM client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                # Fallback to a simple implementation
                pass
        return self._client
    
    def _build_prompt(self, content: str, level: CompressionLevel, context: Optional[Dict]) -> str:
        """Build the summarization prompt based on compression level."""
        base_instructions = """You are a memory compression system for AI agents. 
Your task is to compress the given content while preserving critical information.
Always preserve: named entities (people, organizations, places), decisions made, commitments/promises, key facts, and important dates."""
        
        level_instructions = {
            CompressionLevel.LIGHT: """
Compress lightly - remove redundancy, filler words, and repetition.
Keep all details, examples, and nuance.
Target compression ratio: ~70% of original.""",
            CompressionLevel.MEDIUM: """
Create a structured summary with:
- Key entities mentioned
- Main topics discussed
- Decisions made
- Action items/commitments
- Important facts/figures
Target compression ratio: ~40% of original.""",
            CompressionLevel.AGGRESSIVE: """
Extract ONLY the most critical information:
- Named entities (people, orgs, places)
- Explicit decisions/agreements
- Commitments/promises with deadlines
- Critical facts/numbers
- Unresolved questions
Discard all narrative, examples, and context.
Target compression ratio: ~15% of original.""",
        }
        
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context)}"
        
        return f"""{base_instructions}
{level_instructions.get(level, level_instructions[CompressionLevel.MEDIUM])}
{context_str}

Content to compress:
{content}

Output as JSON with keys: compressed_content, preserved_entities, preserved_decisions, preserved_commitments"""
    
    async def summarize(
        self, 
        content: str, 
        level: CompressionLevel,
        context: Optional[Dict[str, Any]] = None
    ) -> CompressionResult:
        client = await self._get_client()
        
        if client is None:
            # Fallback: simple truncation
            return self._fallback_compress(content, level)
        
        prompt = self._build_prompt(content, level, context)
        
        try:
            response = await client.chat.completions.create(
                model=self.settings.compression.llm_model,
                messages=[
                    {"role": "system", "content": "You are a memory compression system. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=self.settings.compression.max_tokens_per_summary,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return CompressionResult(
                original_content=content,
                compressed_content=result.get("compressed_content", content),
                compression_ratio=len(result.get("compressed_content", content)) / len(content) if content else 1.0,
                level=level,
                preserved_entities=result.get("preserved_entities", []),
                preserved_decisions=result.get("preserved_decisions", []),
                preserved_commitments=result.get("preserved_commitments", []),
                metadata={"model": self.settings.compression.llm_model}
            )
        except Exception as e:
            return self._fallback_compress(content, level)
    
    def _fallback_compress(self, content: str, level: CompressionLevel) -> CompressionResult:
        """Fallback compression without LLM."""
        ratios = {
            CompressionLevel.LIGHT: 0.7,
            CompressionLevel.MEDIUM: 0.4,
            CompressionLevel.AGGRESSIVE: 0.15,
        }
        target_len = int(len(content) * ratios.get(level, 0.4))
        compressed = content[:target_len] + "..." if len(content) > target_len else content
        
        return CompressionResult(
            original_content=content,
            compressed_content=compressed,
            compression_ratio=len(compressed) / len(content) if content else 1.0,
            level=level,
            metadata={"fallback": True}
        )


class HierarchicalSummarizer:
    """Hierarchical summarization across time scales."""
    
    def __init__(self, strategy: SummarizationStrategy, settings=None):
        self.strategy = strategy
        self.settings = settings or get_settings()
        self.levels = self.settings.compression.hierarchical_levels
    
    async def summarize_conversation(
        self,
        messages: List[Dict[str, Any]],
        level: CompressionLevel = CompressionLevel.MEDIUM
    ) -> CompressionResult:
        """Summarize a full conversation."""
        # Combine messages into content
        content_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            content_parts.append(f"[{timestamp}] {role}: {content}")
        
        full_content = "\n".join(content_parts)
        context = {
            "message_count": len(messages),
            "participants": list(set(m.get("role") for m in messages)),
            "time_span": "conversation"
        }
        
        return await self.strategy.summarize(full_content, level, context)
    
    async def create_hierarchical_summaries(
        self,
        memories: List[MemoryRecord],
    ) -> Dict[str, CompressionResult]:
        """Create summaries at multiple time granularities."""
        # Group memories by time period
        by_level = {level: [] for level in self.levels}
        
        for mem in memories:
            # Assign to appropriate levels based on age
            age = utcnow() - (mem.timestamp.replace(tzinfo=timezone.utc) if mem.timestamp.tzinfo is None else mem.timestamp)
            if age < timedelta(hours=1):
                by_level["turn"].append(mem)
            elif age < timedelta(days=1):
                by_level["session"].append(mem)
            elif age < timedelta(days=7):
                by_level["day"].append(mem)
            else:
                by_level["week"].append(mem)
        
        # Create summaries for each level
        results = {}
        for level, level_memories in by_level.items():
            if not level_memories:
                continue
            
            # Convert to message format
            messages = [
                {
                    "role": "assistant" if "assistant" in m.tags else "user",
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in level_memories
            ]
            
            # Use appropriate compression level
            comp_level = {
                "turn": CompressionLevel.LIGHT,
                "session": CompressionLevel.MEDIUM,
                "day": CompressionLevel.MEDIUM,
                "week": CompressionLevel.AGGRESSIVE,
            }.get(level, CompressionLevel.MEDIUM)
            
            results[level] = await self.summarize_conversation(messages, comp_level)
        
        return results


class MemoryCompressor:
    """Main memory compression interface."""
    
    def __init__(self, backend: MemoryBackend, settings=None):
        self.backend = backend
        self.settings = settings or get_settings()
        self.strategy = LLMSummarizationStrategy(settings)
        self.hierarchical = HierarchicalSummarizer(self.strategy, settings)
        self._enabled = self.settings.compression.enabled
    
    async def compress_memory(self, memory: MemoryRecord, level: CompressionLevel = None) -> MemoryRecord:
        """Compress a single memory record."""
        if not self._enabled:
            return memory
        
        level = level or CompressionLevel.MEDIUM
        result = await self.strategy.summarize(memory.content, level)
        
        # Create compressed version
        compressed = MemoryRecord(
            agent_id=memory.agent_id,
            content=result.compressed_content,
            memory_type=memory.memory_type,
            tier=memory.tier,
            importance_score=memory.importance_score,
            tags=memory.tags + ["compressed"],
            metadata={
                **memory.metadata,
                "original_id": memory.id,
                "compression_ratio": result.compression_ratio,
                "compression_level": level.value,
                "preserved_entities": result.preserved_entities,
                "preserved_decisions": result.preserved_decisions,
                "preserved_commitments": result.preserved_commitments,
            },
            compressed=True,
            summary=result.compressed_content,
            parent_id=memory.id,
        )
        
        return compressed
    
    async def compress_batch(
        self, 
        memories: List[MemoryRecord], 
        level: CompressionLevel = None
    ) -> List[MemoryRecord]:
        """Compress multiple memories."""
        results = []
        for mem in memories:
            compressed = await self.compress_memory(mem, level)
            results.append(compressed)
        return results
    
    async def create_hierarchical_summaries(
        self,
        agent_id: str,
        limit: int = 1000
    ) -> Dict[str, CompressionResult]:
        """Create hierarchical summaries for an agent."""
        memories = await self.backend.get_recent(agent_id, limit=limit)
        return await self.hierarchical.create_hierarchical_summaries(memories)
    
    async def auto_compress_old_memories(
        self,
        agent_id: str,
        age_threshold_days: int = 30,
        level: CompressionLevel = CompressionLevel.AGGRESSIVE
    ) -> int:
        """Automatically compress memories older than threshold."""
        cutoff = utcnow() - timedelta(days=age_threshold_days)
        
        # Get old memories
        memories = await self.backend.get_recent(agent_id, limit=10000)
        old_memories = [m for m in memories if (m.timestamp.replace(tzinfo=timezone.utc) if m.timestamp.tzinfo is None else m.timestamp) < cutoff and not m.compressed]
        
        if not old_memories:
            return 0
        
        # Compress them
        compressed = await self.compress_batch(old_memories, level)
        
        # Store compressed versions
        for comp in compressed:
            await self.backend.add(comp)
        
        return len(compressed)


async def create_memory_compressor(
    backend: MemoryBackend,
    settings=None
) -> MemoryCompressor:
    """Factory function to create a memory compressor."""
    return MemoryCompressor(backend, settings)