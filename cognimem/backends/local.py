"""
CogniMem v2.0 - Local Memory Backend Implementations
ChromaDB for vectors, NetworkX for graphs, SentenceTransformers for embeddings.
"""

import asyncio
import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import networkx as nx
from sentence_transformers import SentenceTransformer

from cognimem.backends.base import (
    MemoryBackend, VectorBackend, GraphBackend, EmbeddingBackend,
    MemoryRecord, MemoryQuery, SearchResult
)
from cognimem.config import get_settings


class ChromaVectorBackend(VectorBackend):
    """ChromaDB vector backend implementation."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self._embedding_fn = None
    
    async def initialize(self) -> None:
        self.client = chromadb.PersistentClient(
            path=self.settings.episodic_memory.persist_directory
        )
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.settings.embedding.model
        )
        self.collection = self.client.get_or_create_collection(
            name=self.settings.episodic_memory.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": self.settings.episodic_memory.distance_metric}
        )
    
    async def close(self) -> None:
        # ChromaDB client doesn't need explicit close
        pass
    
    async def add_vectors(
        self, 
        ids: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]]
    ) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    async def search_vectors(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        
        for i, doc in enumerate(docs):
            output.append({
                "id": ids[i],
                "content": doc,
                "metadata": metas[i] if i < len(metas) else {},
                "distance": distances[i] if i < len(distances) else 1.0,
                "similarity": 1.0 - (distances[i] if i < len(distances) else 1.0)
            })
        return output
    
    async def delete_vectors(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)
    
    async def update_vectors(
        self, 
        ids: List[str], 
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        self.collection.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )


class NetworkXGraphBackend(GraphBackend):
    """NetworkX graph backend implementation."""
    
    def __init__(self, settings=None, agent_id: str = "default"):
        self.settings = settings or get_settings()
        self.agent_id = agent_id
        self.graph = nx.Graph()
        self.graph_file = self._get_graph_file()
        self._load()
    
    def _get_graph_file(self) -> str:
        path = Path(self.settings.semantic_memory.persist_directory) / f"{self.agent_id}.{self.settings.semantic_memory.graph_format}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    async def initialize(self) -> None:
        self._load()
    
    async def close(self) -> None:
        self._save()
    
    def _load(self) -> None:
        try:
            if self.settings.semantic_memory.graph_format == "json":
                with open(self.graph_file, "r") as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
            elif self.settings.semantic_memory.graph_format == "graphml":
                self.graph = nx.read_graphml(self.graph_file)
        except (FileNotFoundError, json.JSONDecodeError):
            self.graph = nx.Graph()
    
    def _save(self) -> None:
        try:
            if self.settings.semantic_memory.graph_format == "json":
                data = nx.node_link_data(self.graph)
                with open(self.graph_file, "w") as f:
                    json.dump(data, f)
            elif self.settings.semantic_memory.graph_format == "graphml":
                nx.write_graphml(self.graph, self.graph_file)
        except Exception:
            pass  # Silently fail on save
    
    async def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        self.graph.add_node(node_id, **properties)
        self._save()
    
    async def add_edge(self, source: str, target: str, weight: float = 1.0, properties: Optional[Dict[str, Any]] = None) -> None:
        props = {"weight": weight}
        if properties:
            props.update(properties)
        self.graph.add_edge(source, target, **props)
        self._save()
    
    async def get_neighbors(self, node_id: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        if node_id not in self.graph:
            return []
        
        neighbors = []
        visited = set([node_id])
        current_level = [node_id]
        
        for depth in range(max_depth):
            next_level = []
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                        node_data = self.graph.nodes[neighbor]
                        neighbors.append({
                            "id": neighbor,
                            "depth": depth + 1,
                            **node_data
                        })
            current_level = next_level
            if not current_level:
                break
        
        return neighbors
    
    async def find_paths(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    async def get_subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        subgraph = self.graph.subgraph(node_ids)
        return nx.node_link_data(subgraph)
    
    async def query_by_properties(self, properties: Dict[str, Any], limit: int = 100) -> List[str]:
        results = []
        for node, data in self.graph.nodes(data=True):
            match = all(data.get(k) == v for k, v in properties.items())
            if match:
                results.append(node)
                if len(results) >= limit:
                    break
        return results
    
    async def compute_centrality(self, node_ids: Optional[List[str]] = None) -> Dict[str, float]:
        if node_ids:
            subgraph = self.graph.subgraph(node_ids)
            return nx.degree_centrality(subgraph)
        return nx.degree_centrality(self.graph)
    
    async def cluster_nodes(self, algorithm: str = "louvain") -> Dict[str, int]:
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            return partition
        except ImportError:
            # Fallback to connected components
            clusters = {}
            for i, component in enumerate(nx.connected_components(self.graph)):
                for node in component:
                    clusters[node] = i
            return clusters


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    """SentenceTransformers embedding backend."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None
    
    async def initialize(self) -> None:
        self._model = SentenceTransformer(
            self.settings.embedding.model,
            device=self.settings.embedding.device
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
    
    async def close(self) -> None:
        self._model = None
    
    async def embed_text(self, text: str) -> List[float]:
        if not self._model:
            await self.initialize()
        embedding = self._model.encode(text, normalize_embeddings=self.settings.embedding.normalize_embeddings)
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._model:
            await self.initialize()
        embeddings = self._model.encode(
            texts, 
            batch_size=self.settings.embedding.batch_size,
            normalize_embeddings=self.settings.embedding.normalize_embeddings
        )
        return embeddings.tolist()
    
    async def embed_query(self, query: str) -> List[float]:
        return await self.embed_text(query)
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Initialize lazily
            model = SentenceTransformer(self.settings.embedding.model)
            self._dimension = model.get_sentence_embedding_dimension()
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self.settings.embedding.model


class LocalMemoryBackend(MemoryBackend):
    """Local filesystem-based memory backend using ChromaDB + NetworkX."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.vector_backend = ChromaVectorBackend(settings)
        self.graph_backend = None  # Per-agent, created on demand
        self.embedding_backend = SentenceTransformerEmbeddingBackend(settings)
        self._graph_backends: Dict[str, NetworkXGraphBackend] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.vector_backend.initialize()
        await self.embedding_backend.initialize()
        self._initialized = True
    
    async def close(self) -> None:
        await self.vector_backend.close()
        await self.embedding_backend.close()
        for backend in self._graph_backends.values():
            await backend.close()
        self._graph_backends.clear()
        self._initialized = False
    
    def _get_graph_backend(self, agent_id: str) -> NetworkXGraphBackend:
        if agent_id not in self._graph_backends:
            self._graph_backends[agent_id] = NetworkXGraphBackend(self.settings, agent_id)
        return self._graph_backends[agent_id]
    
    async def health_check(self) -> bool:
        try:
            # Check if we can access the vector backend
            _ = self.vector_backend.client.list_collections()
            return True
        except Exception:
            return False
    
    async def add(self, record: MemoryRecord) -> str:
        await self.initialize()
        
        # Generate embedding
        embedding = await self.embedding_backend.embed_text(record.content)
        record.embedding = embedding
        
        # Store in vector backend
        await self.vector_backend.add_vectors(
            ids=[record.id],
            embeddings=[embedding],
            metadatas=[{
                "agent_id": record.agent_id,
                "content": record.content,
                "memory_type": record.memory_type,
                "tier": record.tier,
                "timestamp": record.timestamp.isoformat(),
                "importance_score": record.importance_score,
                "tags": ",".join(record.tags),
                "version": record.version,
                "parent_id": record.parent_id or "",
                "compressed": record.compressed,
                "summary": record.summary or "",
            }]
        )
        
        # Add to graph if semantic memory
        if record.memory_type == "semantic" or record.tier == "long_term":
            graph = self._get_graph_backend(record.agent_id)
            await graph.add_node(record.id, {
                "content": record.content,
                "tags": record.tags,
                "importance": record.importance_score,
                "type": "fact"
            })
            # Connect to tags
            for tag in record.tags:
                tag_id = f"tag:{tag}"
                await graph.add_node(tag_id, {"type": "concept", "label": tag})
                await graph.add_edge(record.id, tag_id, weight=record.importance_score)
        
        return record.id
    
    async def add_batch(self, records: List[MemoryRecord]) -> List[str]:
        await self.initialize()
        
        # Generate embeddings in batch
        contents = [r.content for r in records]
        embeddings = await self.embedding_backend.embed_batch(contents)
        
        for record, embedding in zip(records, embeddings):
            record.embedding = embedding
        
        # Store in vector backend
        await self.vector_backend.add_vectors(
            ids=[r.id for r in records],
            embeddings=embeddings,
            metadatas=[{
                "agent_id": r.agent_id,
                "content": r.content,
                "memory_type": r.memory_type,
                "tier": r.tier,
                "timestamp": r.timestamp.isoformat(),
                "importance_score": r.importance_score,
                "tags": ",".join(r.tags),
                "version": r.version,
                "parent_id": r.parent_id or "",
                "compressed": r.compressed,
                "summary": r.summary or "",
            } for r in records]
        )
        
        return [r.id for r in records]
    
    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        await self.initialize()
        # Search by ID in vector backend
        results = await self.vector_backend.search_vectors(
            query_embedding=[0.0] * self.embedding_backend.dimension,  # dummy
            top_k=1,
            filters={"id": memory_id}
        )
        if results:
            return self._result_to_record(results[0])
        return None
    
    async def update(self, record: MemoryRecord) -> bool:
        await self.initialize()
        
        # Update embedding if content changed
        embedding = None
        if record.embedding is None:
            embedding = await self.embedding_backend.embed_text(record.content)
            record.embedding = embedding
        
        await self.vector_backend.update_vectors(
            ids=[record.id],
            embeddings=[embedding] if embedding else None,
            metadatas=[{
                "importance_score": record.importance_score,
                "timestamp": record.timestamp.isoformat(),
                "tags": ",".join(record.tags),
                "version": record.version,
                "compressed": record.compressed,
                "summary": record.summary or "",
            }]
        )
        return True
    
    async def delete(self, memory_id: str) -> bool:
        await self.initialize()
        await self.vector_backend.delete_vectors([memory_id])
        return True
    
    async def delete_by_agent(self, agent_id: str, memory_type: Optional[str] = None) -> int:
        await self.initialize()
        filters = {"agent_id": agent_id}
        if memory_type:
            filters["memory_type"] = memory_type
        
        # Get all matching IDs first
        results = await self.vector_backend.search_vectors(
            query_embedding=[0.0] * self.embedding_backend.dimension,
            top_k=10000,
            filters=filters
        )
        
        ids = [r["id"] for r in results]
        if ids:
            await self.vector_backend.delete_vectors(ids)
        return len(ids)
    
    def _result_to_record(self, result: Dict[str, Any]) -> MemoryRecord:
        meta = result.get("metadata", {})
        return MemoryRecord(
            id=result["id"],
            agent_id=meta.get("agent_id", ""),
            content=result["content"],
            memory_type=meta.get("memory_type", "episodic"),
            tier=meta.get("tier", "episodic"),
            timestamp=datetime.fromisoformat(meta.get("timestamp", datetime.utcnow().isoformat())),
            importance_score=meta.get("importance_score", 1.0),
            tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
            version=meta.get("version", 1),
            parent_id=meta.get("parent_id") or None,
            compressed=meta.get("compressed", False),
            summary=meta.get("summary") or None,
        )
    
    async def search(self, query: MemoryQuery) -> List[SearchResult]:
        await self.initialize()
        
        # Build filters
        filters = {"agent_id": query.agent_id}
        if query.memory_types:
            filters["memory_type"] = {"$in": query.memory_types}
        if query.tiers:
            filters["tier"] = {"$in": query.tiers}
        if query.filters:
            filters.update(query.filters)
        
        # Generate query embedding
        query_embedding = await self.embedding_backend.embed_query(query.query)
        
        # Search vector backend
        results = await self.vector_backend.search_vectors(
            query_embedding=query_embedding,
            top_k=query.top_k * 3,  # Get more for reranking
            filters=filters
        )
        
        # Convert to SearchResult with scoring
        search_results = []
        for res in results:
            record = self._result_to_record(res)
            
            # Skip if below minimum importance
            if record.importance_score < query.min_importance:
                continue
            
            similarity = res.get("similarity", 0.5)
            importance = record.importance_score
            
            # Calculate recency score (exponential decay)
            hours_passed = (datetime.utcnow() - record.timestamp).total_seconds() / 3600
            k = self.settings.consolidation.decay_constant_k * (1 + importance)
            recency = importance * (2.71828 ** (-hours_passed / k))
            
            # Final weighted score
            final = (
                similarity * self.settings.retrieval.vector_weight +
                importance * self.settings.retrieval.importance_weight +
                recency * self.settings.retrieval.recency_weight
            )
            
            search_results.append(SearchResult(
                record=record,
                similarity_score=similarity,
                importance_score=importance,
                recency_score=recency,
                final_score=final
            ))
        
        # Sort by final score
        search_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return search_results[:query.top_k]
    
    async def search_by_embedding(
        self, 
        embedding: List[float], 
        agent_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        await self.initialize()
        
        search_filters = {"agent_id": agent_id}
        if filters:
            search_filters.update(filters)
        
        results = await self.vector_backend.search_vectors(
            query_embedding=embedding,
            top_k=top_k,
            filters=search_filters
        )
        
        search_results = []
        for res in results:
            record = self._result_to_record(res)
            similarity = res.get("similarity", 0.5)
            importance = record.importance_score
            
            hours_passed = (datetime.utcnow() - record.timestamp).total_seconds() / 3600
            k = self.settings.consolidation.decay_constant_k * (1 + importance)
            recency = importance * (2.71828 ** (-hours_passed / k))
            
            final = (
                similarity * self.settings.retrieval.vector_weight +
                importance * self.settings.retrieval.importance_weight +
                recency * self.settings.retrieval.recency_weight
            )
            
            search_results.append(SearchResult(
                record=record,
                similarity_score=similarity,
                importance_score=importance,
                recency_score=recency,
                final_score=final
            ))
        
        search_results.sort(key=lambda x: x.final_score, reverse=True)
        return search_results
    
    async def get_by_tags(
        self, 
        agent_id: str, 
        tags: List[str], 
        match_all: bool = False,
        limit: int = 100
    ) -> List[MemoryRecord]:
        await self.initialize()
        
        # Use graph backend to find tagged memories
        graph = self._get_graph_backend(agent_id)
        all_results = []
        
        for tag in tags:
            tag_id = f"tag:{tag}"
            neighbors = await graph.get_neighbors(tag_id, max_depth=1)
            for neighbor in neighbors:
                if neighbor.get("type") == "fact":
                    # Get full record from vector backend
                    record = await self.get(neighbor["id"])
                    if record:
                        all_results.append(record)
        
        if match_all and len(tags) > 1:
            # Filter to only records with ALL tags
            tag_sets = [set(tags)] * len(all_results)  # Simplified
            # In practice, would check each record's tags
            pass
        
        # Sort by importance
        all_results.sort(key=lambda x: x.importance_score, reverse=True)
        return all_results[:limit]
    
    async def get_recent(
        self, 
        agent_id: str, 
        limit: int = 10,
        memory_type: Optional[str] = None,
        tier: Optional[str] = None
    ) -> List[MemoryRecord]:
        await self.initialize()
        
        filters = {"agent_id": agent_id}
        if memory_type:
            filters["memory_type"] = memory_type
        if tier:
            filters["tier"] = tier
        
        # Use dummy embedding to get recent by timestamp
        results = await self.vector_backend.search_vectors(
            query_embedding=[0.0] * self.embedding_backend.dimension,
            top_k=limit * 2,
            filters=filters
        )
        
        records = [self._result_to_record(r) for r in results]
        # Sort by timestamp descending
        records.sort(key=lambda x: x.timestamp, reverse=True)
        return records[:limit]
    
    async def get_by_time_range(
        self,
        agent_id: str,
        start: datetime,
        end: datetime,
        limit: int = 100
    ) -> List[MemoryRecord]:
        await self.initialize()
        
        # ChromaDB doesn't support range queries directly on metadata
        # Fallback: get recent and filter
        results = await self.vector_backend.search_vectors(
            query_embedding=[0.0] * self.embedding_backend.dimension,
            top_k=limit * 5,
            filters={"agent_id": agent_id}
        )
        
        records = []
        for res in results:
            record = self._result_to_record(res)
            if start <= record.timestamp <= end:
                records.append(record)
        
        records.sort(key=lambda x: x.timestamp, reverse=True)
        return records[:limit]
    
    async def promote_to_tier(self, memory_id: str, target_tier: str) -> bool:
        await self.initialize()
        record = await self.get(memory_id)
        if not record:
            return False
        
        old_tier = record.tier
        record.tier = target_tier
        record.memory_type = "semantic" if target_tier in ["long_term", "archival"] else "episodic"
        record.version += 1
        
        await self.update(record)
        
        # Add to graph if promoted to long-term
        if target_tier in ["long_term", "archival"]:
            graph = self._get_graph_backend(record.agent_id)
            await graph.add_node(record.id, {
                "content": record.content,
                "tags": record.tags,
                "importance": record.importance_score,
                "type": "fact"
            })
            for tag in record.tags:
                tag_id = f"tag:{tag}"
                await graph.add_node(tag_id, {"type": "concept", "label": tag})
                await graph.add_edge(record.id, tag_id, weight=record.importance_score)
        
        return True
    
    async def demote_to_tier(self, memory_id: str, target_tier: str) -> bool:
        await self.initialize()
        record = await self.get(memory_id)
        if not record:
            return False
        
        record.tier = target_tier
        record.memory_type = "episodic"
        record.version += 1
        
        await self.update(record)
        return True
    
    async def get_tier_stats(self, agent_id: str) -> Dict[str, int]:
        await self.initialize()
        
        tiers = ["immediate", "short_term", "long_term", "archival"]
        stats = {tier: 0 for tier in tiers}
        
        results = await self.vector_backend.search_vectors(
            query_embedding=[0.0] * self.embedding_backend.dimension,
            top_k=10000,
            filters={"agent_id": agent_id}
        )
        
        for res in results:
            meta = res.get("metadata", {})
            tier = meta.get("tier", "episodic")
            if tier in stats:
                stats[tier] += 1
            else:
                stats["episodic"] = stats.get("episodic", 0) + 1
        
        return stats
    
    async def get_stats(self, agent_id: str) -> Dict[str, Any]:
        await self.initialize()
        
        results = await self.vector_backend.search_vectors(
            query_embedding=[0.0] * self.embedding_backend.dimension,
            top_k=10000,
            filters={"agent_id": agent_id}
        )
        
        total = len(results)
        if total == 0:
            return {
                "total_memories": 0,
                "by_type": {},
                "by_tier": {},
                "avg_importance": 0.0,
                "storage_size_mb": 0.0
            }
        
        by_type = {}
        by_tier = {}
        importance_sum = 0.0
        
        for res in results:
            meta = res.get("metadata", {})
            mem_type = meta.get("memory_type", "unknown")
            tier = meta.get("tier", "unknown")
            importance = meta.get("importance_score", 0.0)
            
            by_type[mem_type] = by_type.get(mem_type, 0) + 1
            by_tier[tier] = by_tier.get(tier, 0) + 1
            importance_sum += importance
        
        return {
            "total_memories": total,
            "by_type": by_type,
            "by_tier": by_tier,
            "avg_importance": importance_sum / total if total > 0 else 0.0,
            "storage_size_mb": 0.0  # Would need to calculate actual disk usage
        }
    
    async def get_importance_distribution(self, agent_id: str) -> Dict[str, int]:
        await self.initialize()
        
        results = await self.vector_backend.search_vectors(
            query_embedding=[0.0] * self.embedding_backend.dimension,
            top_k=10000,
            filters={"agent_id": agent_id}
        )
        
        distribution = {
            "very_high": 0,    # > 0.8
            "high": 0,         # 0.6 - 0.8
            "medium": 0,       # 0.4 - 0.6
            "low": 0,          # 0.2 - 0.4
            "very_low": 0,     # < 0.2
        }
        
        for res in results:
            meta = res.get("metadata", {})
            importance = meta.get("importance_score", 0.0)
            if importance > 0.8:
                distribution["very_high"] += 1
            elif importance > 0.6:
                distribution["high"] += 1
            elif importance > 0.4:
                distribution["medium"] += 1
            elif importance > 0.2:
                distribution["low"] += 1
            else:
                distribution["very_low"] += 1
        
        return distribution