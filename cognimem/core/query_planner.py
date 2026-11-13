"""
CogniMem v2.0 - Query Planning & Decomposition
Intelligent query analysis and multi-step retrieval.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field

from cognimem.config import get_settings
from cognimem.backends.base import MemoryBackend, MemoryQuery, SearchResult, MemoryRecord


class QueryType(str, Enum):
    """Types of queries."""
    SIMPLE = "simple"           # Direct fact lookup
    COMPLEX = "complex"         # Multi-faceted question
    TEMPORAL = "temporal"       # Time-based query
    RELATIONAL = "relational"   # Relationship query
    CAUSAL = "causal"           # Why/how query
    COMPARATIVE = "comparative" # Comparison query
    PROCEDURAL = "procedural"   # How-to query


class QueryIntent(BaseModel):
    """Analyzed query intent."""
    query_type: QueryType
    entities: List[str] = Field(default_factory=list)
    time_references: List[str] = Field(default_factory=list)
    intent_keywords: List[str] = Field(default_factory=list)
    requires_reasoning: bool = False
    suggested_sub_queries: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class SubQuery(BaseModel):
    """A decomposed sub-query."""
    id: str
    query: str
    query_type: QueryType
    priority: int = 1
    dependencies: List[str] = Field(default_factory=list)  # Other sub-query IDs
    expected_result_type: str = "facts"
    filters: Dict[str, Any] = Field(default_factory=dict)


class QueryPlan(BaseModel):
    """Complete query execution plan."""
    original_query: str
    intent: QueryIntent
    sub_queries: List[SubQuery] = Field(default_factory=list)
    fusion_strategy: str = "rrf"  # rrf, weighted, max, concatenate
    estimated_steps: int = 1
    parallelizable: bool = True


class QueryAnalyzer(ABC):
    """Abstract query analyzer."""
    
    @abstractmethod
    async def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryIntent:
        pass
    
    @abstractmethod
    async def decompose(self, intent: QueryIntent) -> List[SubQuery]:
        pass


class LLMQueryAnalyzer(QueryAnalyzer):
    """LLM-based query analysis and decomposition."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client = None
    
    async def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                pass
        return self._client
    
    def _build_analysis_prompt(self, query: str, context: Optional[Dict]) -> str:
        context_str = f"\nContext: {json.dumps(context)}" if context else ""
        
        return f"""Analyze the following query and return a JSON object with:
- query_type: one of [simple, complex, temporal, relational, causal, comparative, procedural]
- entities: list of named entities mentioned
- time_references: list of temporal expressions
- intent_keywords: key intent signals
- requires_reasoning: boolean
- suggested_sub_queries: list of simpler sub-queries
- confidence: 0.0-1.0

Query: {query}
{context_str}

Output only valid JSON."""
    
    def _build_decomposition_prompt(self, intent: QueryIntent) -> str:
        return f"""Decompose this query intent into atomic sub-queries.
For each sub-query, provide:
- id: unique identifier
- query: the sub-query text
- query_type: type classification
- priority: 1-5 (1=highest)
- dependencies: list of other sub-query IDs this depends on
- expected_result_type: what kind of result (facts, entities, timeline, etc.)
- filters: any metadata filters to apply

Intent: {intent.model_dump_json()}

Output as JSON array of sub-query objects."""
    
    async def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryIntent:
        client = await self._get_client()
        
        if client is None:
            return self._fallback_analyze(query)
        
        prompt = self._build_analysis_prompt(query, context)
        
        try:
            response = await client.chat.completions.create(
                model=self.settings.query_planner.llm_model,
                messages=[
                    {"role": "system", "content": "You are a query analyzer. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return QueryIntent(**result)
        except Exception:
            return self._fallback_analyze(query)
    
    def _fallback_analyze(self, query: str) -> QueryIntent:
        """Simple heuristic-based analysis."""
        query_lower = query.lower()
        
        # Detect query type
        if any(w in query_lower for w in ["when", "date", "time", "yesterday", "last week", "timeline"]):
            qtype = QueryType.TEMPORAL
        elif any(w in query_lower for w in ["why", "how", "cause", "reason", "because"]):
            qtype = QueryType.CAUSAL
        elif any(w in query_lower for w in ["compare", "versus", "vs", "difference", "similar"]):
            qtype = QueryType.COMPARATIVE
        elif any(w in query_lower for w in ["how to", "steps", "process", "procedure"]):
            qtype = QueryType.PROCEDURAL
        elif any(w in query_lower for w in ["relationship", "related", "connected", "associated"]):
            qtype = QueryType.RELATIONAL
        elif len(query.split()) > 10 or "?" in query and query.count("?") > 1:
            qtype = QueryType.COMPLEX
        else:
            qtype = QueryType.SIMPLE
        
        # Extract simple entities (capitalized words)
        entities = [w.strip(".,?!") for w in query.split() if w[0].isupper() and len(w) > 2]
        
        # Simple time references
        time_refs = [w for w in query_lower.split() if w in ["yesterday", "today", "tomorrow", "week", "month", "year", "ago", "last", "next"]]
        
        return QueryIntent(
            query_type=qtype,
            entities=entities[:5],
            time_references=time_refs,
            intent_keywords=query_lower.split()[:10],
            requires_reasoning=qtype in [QueryType.CAUSAL, QueryType.COMPARATIVE, QueryType.PROCEDURAL],
            suggested_sub_queries=[],
            confidence=0.6
        )
    
    async def decompose(self, intent: QueryIntent) -> List[SubQuery]:
        client = await self._get_client()
        
        if client is None or intent.query_type == QueryType.SIMPLE:
            return self._fallback_decompose(intent)
        
        prompt = self._build_decomposition_prompt(intent)
        
        try:
            response = await client.chat.completions.create(
                model=self.settings.query_planner.llm_model,
                messages=[
                    {"role": "system", "content": "You are a query decomposer. Output only valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if isinstance(result, dict) and "sub_queries" in result:
                result = result["sub_queries"]
            
            return [SubQuery(**sq) for sq in result[:self.settings.query_planner.max_sub_queries]]
        except Exception:
            return self._fallback_decompose(intent)
    
    def _fallback_decompose(self, intent: QueryIntent) -> List[SubQuery]:
        """Simple fallback decomposition."""
        if intent.query_type == QueryType.SIMPLE:
            return [SubQuery(
                id="sq_1",
                query=intent.intent_keywords[0] if intent.intent_keywords else "query",
                query_type=intent.query_type,
                priority=1
            )]
        
        # Create basic sub-queries from intent
        sub_queries = []
        for i, entity in enumerate(intent.entities[:3]):
            sub_queries.append(SubQuery(
                id=f"sq_{i+1}",
                query=f"Information about {entity}",
                query_type=QueryType.SIMPLE,
                priority=i+1,
                expected_result_type="entity_info"
            ))
        
        if intent.time_references:
            sub_queries.append(SubQuery(
                id=f"sq_time",
                query=f"Timeline of events around {' '.join(intent.time_references)}",
                query_type=QueryType.TEMPORAL,
                priority=len(sub_queries)+1,
                expected_result_type="timeline"
            ))
        
        return sub_queries[:self.settings.query_planner.max_sub_queries]


class QueryPlanner:
    """Main query planning interface."""
    
    def __init__(self, analyzer: QueryAnalyzer, settings=None):
        self.analyzer = analyzer
        self.settings = settings or get_settings()
    
    async def create_plan(
        self, 
        query: str, 
        agent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Create a complete query execution plan."""
        # Analyze intent
        intent = await self.analyzer.analyze(query, context)
        
        # Decompose into sub-queries
        sub_queries = await self.analyzer.decompose(intent)
        
        # Determine fusion strategy
        fusion = self.settings.retrieval.fusion_strategy if hasattr(self.settings.retrieval, 'fusion_strategy') else "rrf"
        if intent.query_type in [QueryType.COMPLEX, QueryType.CAUSAL, QueryType.COMPARATIVE]:
            fusion = "weighted"
        
        return QueryPlan(
            original_query=query,
            intent=intent,
            sub_queries=sub_queries,
            fusion_strategy=fusion,
            estimated_steps=len(sub_queries),
            parallelizable=len(sub_queries) > 1 and all(not sq.dependencies for sq in sub_queries)
        )
    
    async def execute_plan(
        self,
        plan: QueryPlan,
        backend: MemoryBackend,
        agent_id: str
    ) -> List[SearchResult]:
        """Execute a query plan against the backend."""
        all_results = []
        
        if plan.parallelizable and self.settings.query_planner.parallel_retrieval:
            # Execute all sub-queries in parallel
            tasks = []
            for sq in plan.sub_queries:
                query = MemoryQuery(
                    query=sq.query,
                    agent_id=agent_id,
                    top_k=self.settings.retrieval.top_k,
                    filters=sq.filters,
                )
                tasks.append(backend.search(query))
            
            results_lists = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, results in enumerate(results_lists):
                if isinstance(results, Exception):
                    continue
                # Tag results with sub-query ID
                for r in results:
                    r.record.metadata["sub_query_id"] = plan.sub_queries[i].id
                all_results.extend(results)
        else:
            # Execute sequentially (respecting dependencies)
            completed = set()
            for sq in plan.sub_queries:
                if sq.dependencies and not all(d in completed for d in sq.dependencies):
                    # Skip for now, would need topological sort
                    continue
                
                query = MemoryQuery(
                    query=sq.query,
                    agent_id=agent_id,
                    top_k=self.settings.retrieval.top_k,
                    filters=sq.filters,
                )
                results = await backend.search(query)
                
                for r in results:
                    r.record.metadata["sub_query_id"] = sq.id
                all_results.extend(results)
                completed.add(sq.id)
        
        # Fuse results
        fused = self._fuse_results(all_results, plan.fusion_strategy)
        return fused[:self.settings.retrieval.top_k]
    
    def _fuse_results(
        self, 
        results: List[SearchResult], 
        strategy: str
    ) -> List[SearchResult]:
        """Fuse results from multiple sub-queries."""
        if not results:
            return []
        
        # Group by memory ID
        by_id = {}
        for r in results:
            mem_id = r.record.id
            if mem_id not in by_id:
                by_id[mem_id] = []
            by_id[mem_id].append(r)
        
        fused = []
        for mem_id, group in by_id.items():
            if strategy == "rrf":
                # Reciprocal Rank Fusion
                score = sum(1.0 / (i + 1 + 60) for i, _ in enumerate(group))
            elif strategy == "max":
                score = max(r.final_score for r in group)
            elif strategy == "weighted":
                score = sum(r.final_score for r in group) / len(group)
            else:  # concatenate
                score = group[0].final_score
            
            # Use the best result as base
            best = max(group, key=lambda x: x.final_score)
            fused.append(SearchResult(
                record=best.record,
                similarity_score=best.similarity_score,
                importance_score=best.importance_score,
                recency_score=best.recency_score,
                final_score=score,
                matched_fields=list(set().union(*[set(r.matched_fields) for r in group]))
            ))
        
        fused.sort(key=lambda x: x.final_score, reverse=True)
        return fused


class QueryRewriter:
    """Query rewriting for improved retrieval."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client = None
    
    async def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                pass
        return self._client
    
    async def rewrite(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate alternative query formulations."""
        client = await self._get_client()
        
        if client is None:
            return self._fallback_rewrite(query)
        
        context_str = f"\nContext: {json.dumps(context)}" if context else ""
        
        prompt = f"""Generate 3 alternative formulations of this query for better memory retrieval.
Each should capture the same intent but use different wording.
Include: synonyms, related concepts, more specific/broader terms.

Original query: {query}
{context_str}

Output as JSON array of strings."""
        
        try:
            response = await client.chat.completions.create(
                model=self.settings.query_planner.llm_model,
                messages=[
                    {"role": "system", "content": "You are a query rewriter. Output only valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if isinstance(result, dict) and "queries" in result:
                return result["queries"]
            elif isinstance(result, list):
                return result
        except Exception:
            pass
        
        return self._fallback_rewrite(query)
    
    def _fallback_rewrite(self, query: str) -> List[str]:
        """Simple fallback rewrites."""
        return [
            query,
            query.replace("what", "information about"),
            query.replace("how", "process for"),
            query.replace("why", "reason for"),
        ]


async def create_query_planner(settings=None) -> QueryPlanner:
    """Factory function to create a query planner."""
    analyzer = LLMQueryAnalyzer(settings)
    return QueryPlanner(analyzer, settings)