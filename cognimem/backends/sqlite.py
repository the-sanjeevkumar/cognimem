"""
CogniMem v2.0 - SQLite Memory Backend (zero-dependency, production-safe default)

Implements the full MemoryBackend contract on top of a single SQLite database:
- Exact CRUD by ID (no vector hacks for get/stats/scan operations)
- Real metadata filters: tags, time ranges, tiers, importance
- Proper upsert semantics with version bumps
- Vector search via cosine similarity over stored embeddings (works with any
  EmbeddingBackend, including the lightweight built-in ones)
- Optional FTS5 keyword index for hybrid lexical + vector retrieval
- WAL mode + busy timeout for concurrent readers/writers
"""

import asyncio
import json
import math
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from cognimem.backends.base import (
    MemoryBackend,
    EmbeddingBackend,
    MemoryRecord,
    MemoryQuery,
    SearchResult,
)
from cognimem.config import get_settings


_WORD_RE = re.compile(r"[\w\-]+")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    """Attach UTC to naive timestamps so comparisons never crash."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


class HashingEmbeddingBackend(EmbeddingBackend):
    """
    Dependency-free embedding backend using hashed character n-grams with
    L2 normalization. Not semantically deep, but deterministic, instant and
    good enough for lexical-style similarity when no model is available.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self._dimension
        lowered = text.lower()
        # word unigrams + character trigrams
        tokens = _WORD_RE.findall(lowered)
        for tok in tokens:
            vec[hash(tok) % self._dimension] += 1.0
        padded = f" {lowered} "
        for i in range(len(padded) - 2):
            vec[hash(padded[i:i + 3]) % self._dimension] += 0.5
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed_text(t) for t in texts]

    async def embed_query(self, query: str) -> List[float]:
        return await self.embed_text(query)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "hashing-ngram-384"


class SQLiteMemoryBackend(MemoryBackend):
    """
    Self-contained memory backend backed by SQLite.

    Advantages over the ChromaDB-based local backend:
    - No heavy ML dependencies required to run or test
    - Exact `get`/`update`/`delete`/scans instead of approximate vector hacks
    - Real tag/time/tier filtering at the storage layer
    - Atomic writes, schema migrations, and one file per agent namespace
    """

    _FTS_TOKENIZER_EXPR = "unicode61"

    def __init__(self, settings=None, db_path: Optional[str] = None,
                 embedding_backend: Optional[EmbeddingBackend] = None):
        self.settings = settings or get_settings()
        base_dir = db_path or os.path.join(
            self.settings.episodic_memory.persist_directory, "sqlite"
        )
        os.makedirs(base_dir, exist_ok=True)
        self.db_path = os.path.join(base_dir, f"{self.settings.agent_id}.db")
        self.embedding_backend = embedding_backend or HashingEmbeddingBackend()
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False
        self._fts_enabled = False

    # --- lifecycle -----------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)
        self._initialized = True

    def _init_sync(self) -> None:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                tier TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                importance_score REAL NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                version INTEGER NOT NULL DEFAULT 1,
                parent_id TEXT,
                compressed INTEGER NOT NULL DEFAULT 0,
                summary TEXT,
                embedding TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_mem_tier ON memories(agent_id, tier);
            CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(agent_id, memory_type);
            CREATE INDEX IF NOT EXISTS idx_mem_time ON memories(agent_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_mem_importance ON memories(agent_id, importance_score);
            """
        )
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content, tags, content='memories', content_rowid='rowid',
                    tokenize='{}'
                );
                """.format(self._FTS_TOKENIZER_EXPR)
            )
            conn.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (new.rowid, new.content, new.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.rowid, old.content, old.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.rowid, old.content, old.tags);
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (new.rowid, new.content, new.tags);
                END;
                """
            )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False
        conn.commit()
        self._conn = conn

    async def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
        self._initialized = False

    async def health_check(self) -> bool:
        try:
            await self.initialize()
            with self._lock:
                self._conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    # --- serialization helpers ----------------------------------------

    def _row_to_record(self, row: Any, include_embedding: bool = False) -> MemoryRecord:
        embedding = None
        if include_embedding and row["embedding"]:
            embedding = json.loads(row["embedding"])
        return MemoryRecord(
            id=row["id"],
            agent_id=row["agent_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            tier=row["tier"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            importance_score=row["importance_score"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            version=row["version"],
            parent_id=row["parent_id"],
            compressed=bool(row["compressed"]),
            summary=row["summary"],
            embedding=embedding,
        )

    def _record_to_params(self, record: MemoryRecord) -> tuple:
        return (
            record.id,
            record.agent_id,
            record.content,
            record.memory_type,
            record.tier,
            record.timestamp.isoformat(),
            float(record.importance_score),
            json.dumps(record.tags),
            json.dumps(record.metadata),
            record.version,
            record.parent_id,
            1 if record.compressed else 0,
            record.summary,
            json.dumps(record.embedding) if record.embedding else None,
        )

    UPSERT_SQL = """
        INSERT INTO memories (id, agent_id, content, memory_type, tier, timestamp,
                              importance_score, tags, metadata, version, parent_id,
                              compressed, summary, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            content=excluded.content,
            memory_type=excluded.memory_type,
            tier=excluded.tier,
            timestamp=excluded.timestamp,
            importance_score=excluded.importance_score,
            tags=excluded.tags,
            metadata=excluded.metadata,
            version=excluded.version,
            parent_id=excluded.parent_id,
            compressed=excluded.compressed,
            summary=excluded.summary,
            embedding=excluded.embedding
    """

    # --- CRUD -----------------------------------------------------------

    async def add(self, record: MemoryRecord) -> str:
        await self.initialize()
        if record.embedding is None:
            record.embedding = await self.embedding_backend.embed_text(record.content)
        with self._lock:
            self._conn.execute(self.UPSERT_SQL, self._record_to_params(record))
            self._conn.commit()
        return record.id

    async def add_batch(self, records: List[MemoryRecord]) -> List[str]:
        await self.initialize()
        if records:
            embeddings = await self.embedding_backend.embed_batch([r.content for r in records])
            for r, e in zip(records, embeddings):
                r.embedding = e
            with self._lock:
                self._conn.executemany(
                    self.UPSERT_SQL, [self._record_to_params(r) for r in records]
                )
                self._conn.commit()
        return [r.id for r in records]

    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        await self.initialize()
        with self._lock:
            cur = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cur.fetchone()
        return self._row_to_record(row) if row else None

    async def update(self, record: MemoryRecord) -> bool:
        await self.initialize()
        with self._lock:
            cur = self._conn.execute("SELECT version FROM memories WHERE id = ?", (record.id,))
            row = cur.fetchone()
            if row is None:
                return False
            record.version = row["version"] + 1
            if record.embedding is None:
                record.embedding = await self.embedding_backend.embed_text(record.content)
            self._conn.execute(self.UPSERT_SQL, self._record_to_params(record))
            self._conn.commit()
        return True

    async def delete(self, memory_id: str) -> bool:
        await self.initialize()
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()
        return cur.rowcount > 0

    async def delete_by_agent(self, agent_id: str, memory_type: Optional[str] = None) -> int:
        await self.initialize()
        sql = "DELETE FROM memories WHERE agent_id = ?"
        params: List[Any] = [agent_id]
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
        return cur.rowcount

    # --- scanning helpers ------------------------------------------------

    def _scan(self, where: str, params: List[Any], order: str = "timestamp DESC",
              limit: Optional[int] = None) -> List[MemoryRecord]:
        sql = f"SELECT * FROM memories WHERE {where} ORDER BY {order}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    # --- scoring ----------------------------------------------------------

    def _score(self, record: MemoryRecord, similarity: float) -> SearchResult:
        importance = record.importance_score
        hours_passed = max(0.0, (_utcnow() - _aware(record.timestamp)).total_seconds() / 3600.0)
        k = self.settings.consolidation.decay_constant_k * (1.0 + importance)
        recency = importance * math.exp(-hours_passed / k)
        final = (
            similarity * self.settings.retrieval.vector_weight
            + importance * self.settings.retrieval.importance_weight
            + recency * self.settings.retrieval.recency_weight
        )
        matched = []
        if similarity >= self.settings.retrieval.min_similarity_threshold:
            matched.append("vector")
        return SearchResult(
            record=record,
            similarity_score=similarity,
            importance_score=importance,
            recency_score=recency,
            final_score=final,
            matched_fields=matched,
        )

    # --- queries ------------------------------------------------------------

    async def search(self, query: MemoryQuery) -> List[SearchResult]:
        await self.initialize()

        where = ["agent_id = ?"]
        params: List[Any] = [query.agent_id]
        if query.memory_types:
            where.append(f"memory_type IN ({','.join('?' * len(query.memory_types))})")
            params.extend(query.memory_types)
        if query.tiers:
            tiers = [t if isinstance(t, str) else t.value for t in query.tiers]
            where.append(f"tier IN ({','.join('?' * len(tiers))})")
            params.extend(tiers)
        if query.min_importance > 0:
            where.append("importance_score >= ?")
            params.append(query.min_importance)
        if query.time_range:
            start, end = query.time_range
            where.append("timestamp BETWEEN ? AND ?")
            params.extend([_aware(start).isoformat(), _aware(end).isoformat()])

        candidates = self._scan(" AND ".join(where), params, limit=max(query.top_k * 10, 50))
        if not candidates:
            return []

        q_emb = await self.embedding_backend.embed_query(query.query)

        results: List[SearchResult] = []
        for record in candidates:
            emb = record.embedding
            if emb:
                sim = _cosine(q_emb, emb)
            else:
                sim = 0.0
            results.append(self._score(record, sim))

        # Boost records matching the FTS keyword index (hybrid lexical signal)
        if getattr(self, "_fts_enabled", False):
            tokens = _WORD_RE.findall(query.query.lower())
            if tokens:
                match_expr = " OR ".join(f'"{t}"' for t in tokens[:10])
                try:
                    with self._lock:
                        fts_rows = self._conn.execute(
                            "SELECT m.id FROM memories_fts f "
                            "JOIN memories m ON m.rowid = f.rowid "
                            "WHERE memories_fts MATCH ? AND m.agent_id = ? LIMIT ?",
                            (match_expr, query.agent_id, len(candidates) * 2),
                        ).fetchall()
                    fts_ids = {r["id"] for r in fts_rows}
                    for r in results:
                        if r.record.id in fts_ids:
                            r.similarity_score = min(1.0, r.similarity_score + 0.15)
                            r.final_score += 0.15 * self.settings.retrieval.vector_weight
                            if "keyword" not in r.matched_fields:
                                r.matched_fields.append("keyword")
                except sqlite3.OperationalError:
                    pass

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[: query.top_k]

    async def search_by_embedding(
        self,
        embedding: List[float],
        agent_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        await self.initialize()
        where = ["agent_id = ?"]
        params: List[Any] = [agent_id]
        filters = filters or {}
        if "tier" in filters:
            where.append("tier = ?")
            params.append(filters["tier"])
        if "memory_type" in filters:
            where.append("memory_type = ?")
            params.append(filters["memory_type"])
        candidates = self._scan(" AND ".join(where), params, limit=max(top_k * 10, 50))

        results = []
        for record in candidates:
            sim = _cosine(embedding, record.embedding) if record.embedding else 0.0
            results.append(self._score(record, sim))
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]

    async def get_by_tags(
        self,
        agent_id: str,
        tags: List[str],
        match_all: bool = False,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        await self.initialize()
        # Fetch generously, then filter exactly on the JSON tag list.
        candidates = self._scan("agent_id = ?", [agent_id], limit=None)
        wanted = set(tags)
        out = []
        for record in candidates:
            have = set(record.tags)
            if (wanted <= have) if match_all else (wanted & have):
                out.append(record)
        out.sort(key=lambda r: r.importance_score, reverse=True)
        return out[:limit]

    async def get_recent(
        self,
        agent_id: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> List[MemoryRecord]:
        await self.initialize()
        where = ["agent_id = ?"]
        params: List[Any] = [agent_id]
        if memory_type:
            where.append("memory_type = ?")
            params.append(memory_type)
        if tier:
            where.append("tier = ?")
            params.append(getattr(tier, "value", tier))
        tier_value = tier
        return self._scan(
            " AND ".join(where), params,
            order=f"julianday(timestamp) DESC", limit=limit,
        )

    async def get_by_time_range(
        self,
        agent_id: str,
        start: datetime,
        end: datetime,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        await self.initialize()
        return self._scan(
            "agent_id = ? AND timestamp BETWEEN ? AND ?",
            [agent_id, _aware(start).isoformat(), _aware(end).isoformat()],
            limit=limit,
        )

    # --- tier operations ---------------------------------------------------

    async def promote_to_tier(self, memory_id: str, target_tier: str) -> bool:
        await self.initialize()
        target = getattr(target_tier, "value", target_tier)
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE memories
                SET tier = ?,
                    memory_type = CASE WHEN ? IN ('long_term','archival')
                                       THEN 'semantic' ELSE 'episodic' END,
                    version = version + 1
                WHERE id = ?
                """,
                (target, target, memory_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    async def demote_to_tier(self, memory_id: str, target_tier: str) -> bool:
        await self.initialize()
        target = getattr(target_tier, "value", target_tier)
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE memories
                SET tier = ?,
                    memory_type = CASE WHEN ? IN ('long_term','archival')
                                       THEN 'semantic' ELSE 'episodic' END,
                    version = version + 1
                WHERE id = ?
                """,
                (target, target, memory_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    async def get_tier_stats(self, agent_id: str) -> Dict[str, int]:
        await self.initialize()
        with self._lock:
            rows = self._conn.execute(
                "SELECT tier, COUNT(*) AS c FROM memories WHERE agent_id = ? GROUP BY tier",
                (agent_id,),
            ).fetchall()
        stats = {"immediate": 0, "short_term": 0, "long_term": 0, "archival": 0}
        for row in rows:
            stats[row["tier"]] = stats.get(row["tier"], 0) + row["c"]
        return stats

    # --- analytics -----------------------------------------------------------

    async def get_stats(self, agent_id: str) -> Dict[str, Any]:
        await self.initialize()
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE agent_id = ?", (agent_id,)
            ).fetchone()["c"]
            avg_imp = self._conn.execute(
                "SELECT AVG(importance_score) AS a FROM memories WHERE agent_id = ?", (agent_id,)
            ).fetchone()["a"]
            by_type_rows = self._conn.execute(
                "SELECT memory_type, COUNT(*) AS c FROM memories WHERE agent_id = ? GROUP BY memory_type",
                (agent_id,),
            ).fetchall()
            by_tier = await self.get_tier_stats(agent_id)
        size_mb = 0.0
        try:
            size_mb = round(os.path.getsize(self.db_path) / (1024 * 1024), 3)
        except OSError:
            pass
        return {
            "total_memories": total,
            "by_type": {r["memory_type"]: r["c"] for r in by_type_rows},
            "by_tier": by_tier,
            "avg_importance": round(avg_imp, 4) if avg_imp is not None else 0.0,
            "storage_size_mb": size_mb,
        }

    async def get_importance_distribution(self, agent_id: str) -> Dict[str, int]:
        await self.initialize()
        buckets = {
            "very_high": (0.8, 1.01),
            "high": (0.6, 0.8),
            "medium": (0.4, 0.6),
            "low": (0.2, 0.4),
            "very_low": (-1.0, 0.2),
        }
        distribution = {k: 0 for k in buckets}
        with self._lock:
            rows = self._conn.execute(
                "SELECT importance_score FROM memories WHERE agent_id = ?", (agent_id,)
            ).fetchall()
        for row in rows:
            score = row["importance_score"]
            for name, (lo, hi) in buckets.items():
                if lo < score <= hi or (name == "very_low" and score <= 0.2):
                    distribution[name] += 1
                    break
        return distribution


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
