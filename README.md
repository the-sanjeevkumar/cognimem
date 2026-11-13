---
# CogniMem v2.0: Advanced Cognitive Memory for AI Agents

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)]()

---

Stop giving your AI agents amnesia or infinite, noisy context windows. **CogniMem** is an open-source, biologically inspired memory architecture that allows AI agents to remember, forget, and learn just like humans do.

Unlike static RAG or simple sliding-window contexts, CogniMem implements a **hierarchical memory system** governed by background consolidation, compression, and proactive retrieval engines. It requires only **minor code changes** to integrate with almost any existing AI agent framework.

## 🚀 What's New in v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Memory Tiers | 3 (Working, Episodic, Semantic) | **4 (Immediate, Short-term, Long-term, Archival)** |
| Retrieval | Hybrid vector + importance | **Query planning + decomposition + cross-session** |
| Memory Management | Ebbinghaus decay | **Compression + summarization + auto-tiering** |
| Proactivity | None | **Context-aware + temporal + pattern-based prefetch** |
| Multi-Agent | Basic agent_id isolation | **Shared spaces + RBAC + conflict resolution** |
| Sessions | None | **Full lifecycle + bridging + cross-session recall** |
| Architecture | Sync, monolithic | **Async-first + pluggable backends + CLI** |

---

## ✨ Key Features

### 🧠 Hierarchical Memory Tiers
```
Immediate (Working) → Short-term (Episodic) → Long-term (Semantic) → Archival (Cold)
     10 items              1,000 items           100,000 items         1,000,000 items
     5 min TTL             7 day TTL             1 year TTL            10 year TTL
```
- Automatic promotion/demotion based on importance and age
- Configurable capacity, TTL, and retrieval latency per tier
- Background rebalancing prevents tier overflow

### 🗜️ LLM-Based Memory Compression
- **Hierarchical summarization**: turn → session → day → week
- Preserves entities, decisions, commitments automatically
- Configurable compression levels (light/medium/aggressive)
- Automatic compression of old memories

### 🎯 Intelligent Query Planning
- LLM-based query analysis and decomposition
- Parallel sub-query execution with result fusion
- Reciprocal Rank Fusion (RRF), weighted, and max strategies
- Query rewriting for improved retrieval

### ⚡ Proactive/Anticipatory Retrieval
- **Context-aware**: Prefetches on goal/topic/entity changes
- **Temporal**: Daily/weekly/monthly scheduled summaries
- **Pattern-based**: Learns query co-occurrence patterns
- Background prefetch with caching

### 🤝 Multi-Agent Shared Memory
- Shared memory spaces with namespaces (personal/team/org/public)
- Role-based access control (read/write/admin)
- Conflict resolution strategies (last-write-wins/merge/manual)
- Full audit trail and version history

### 📋 Session Management
- Full session lifecycle (active → paused → ended → archived)
- Automatic session summarization on end
- Context bridging between related sessions
- Cross-session retrieval with source attribution

### 🔌 Pluggable Architecture
- Abstract backends for vectors (ChromaDB, Pinecone, Weaviate)
- Graph backends (NetworkX, Neo4j, Kuzu)
- Embedding backends (SentenceTransformers, OpenAI, Cohere)
- Storage backends (local, Redis, PostgreSQL, S3)

---

## 📦 Installation

```bash
# Using uv (recommended)
uv pip install cognimem

# Or from source
git clone https://github.com/your-org/cognimem.git
cd cognimem
uv pip install -e .
```

Requires Python 3.10+.

---

## 🚀 Quick Start

```python
from cognimem import create_cognimem
from cognimem.core.hierarchical import MemoryTier

async def main():
    # Create and initialize
    async with create_cognimem(agent_id="research_assistant") as mem:
        
        # Remember with automatic tier placement
        await mem.remember(
            content="The user prefers Python over Java for backend development.",
            importance=0.9,
            tags=["user_preference", "python", "backend"]
        )
        
        await mem.remember(
            content="We discussed deploying the app using Docker yesterday.",
            importance=0.5,
            tags=["project", "docker"]
        )
        
        # Recall with intelligent query planning
        results = await mem.recall(
            query="What programming language does the user like?",
            top_k=3,
            use_query_planner=True
        )
        
        # Tier-specific recall
        results = await mem.recall(
            query="deployment",
            top_k=2,
            tiers=[MemoryTier.LONG_TERM]
        )
        
        # Start a session for conversation tracking
        session_id = await mem.start_session(tags=["meeting", "planning"])
        
        # ... conversation happens ...
        
        # End session with automatic summarization
        await mem.end_session()
        
        # Proactive memory - context-aware prefetch
        await mem.on_context_change({
            "current_goal": "Deploy the application",
            "current_topic": "database selection"
        })

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## 🖥️ Command Line Interface

```bash
# Remember a memory
cognimem remember "User prefers Python" --importance 0.9 --tags python preference

# Recall memories
cognimem recall "What does the user like?" --top-k 5

# Show statistics
cognimem stats

# Tier statistics
cognimem tiers

# Compress old memories
cognimem compress --max-age-days 30 --level aggressive

# Session management
cognimem session start --tags meeting
cognimem session list --status active
cognimem session end

# Shared memory
cognimem shared create "Project Team" --namespace team
cognimem shared remember <space_id> "Team decision: Use Redis"
cognimem shared recall <space_id> "caching decision"

# Proactive triggers
cognimem proactive context current_goal "Deploy application"
cognimem proactive entity "PostgreSQL"
cognimem proactive decision "Which database to use?"
```

---

## ⚙️ Configuration

CogniMem uses `config.yaml` with environment variable overrides (`COGNIMEM_` prefix):

```yaml
agent:
  default_id: "my_agent"

memory:
  working_memory_limit: 10
  consolidation_interval_hours: 6
  decay_constant_k: 24
  promotion_threshold: 0.7
  pruning_threshold: 0.1

embedding:
  model: "all-MiniLM-L6-v2"

retrieval:
  vector_weight: 0.6
  importance_weight: 0.3
  recency_weight: 0.1
  top_k: 5

compression:
  enabled: true
  llm_model: "gpt-4o-mini"
  max_tokens_per_summary: 500

query_planner:
  enabled: true
  llm_model: "gpt-4o-mini"
  max_sub_queries: 5

multi_agent:
  enabled: false
  namespace_prefix: "team_"
```

Environment variables override nested config with `__` delimiter:
```bash
export COGNIMEM_EMBEDDING__MODEL="text-embedding-3-small"
export COGNIMEM_RETRIEVAL__TOP_K=10
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CogniMem SDK                           │
├─────────────────────────────────────────────────────────────┤
│  HierarchicalMemory  │  Compression  │  QueryPlanner        │
│  ProactiveMemory     │  MultiAgent   │  SessionManager      │
├─────────────────────────────────────────────────────────────┤
│                    Abstract Backends                        │
│  MemoryBackend  │  VectorBackend  │  GraphBackend          │
│  EmbeddingBackend                                                          │
├─────────────────────────────────────────────────────────────┤
│  Local (ChromaDB + NetworkX + SentenceTransformers)         │
│  Remote (Pinecone/Weaviate + Neo4j/Kuzu + OpenAI/Cohere)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Advanced Usage

### Custom Backend
```python
from cognimem.backends.base import MemoryBackend, VectorBackend

class MyVectorBackend(VectorBackend):
    async def initialize(self): ...
    async def add_vectors(self, ids, embeddings, metadatas): ...
    async def search_vectors(self, query_embedding, top_k, filters): ...

# Use with CogniMem
from cognimem.backends.local import LocalMemoryBackend
# Or plug in your custom backend
```

### Multi-Agent Team Memory
```python
# Create team space
space = await mem.create_shared_space("Engineering Team", namespace="team")

# Add members with permissions
await mem.multiagent.add_member(space['space_id'], "agent_alice", PermissionLevel.WRITE, "agent_bob")

# Shared remember/recall
await mem.remember_shared(space['space_id'], "Decision: Use PostgreSQL", importance=0.9)
results = await mem.recall_shared(space['space_id'], "database decision")
```

### Cross-Session Continuity
```python
# Session 1
session1 = await mem.start_session()
await mem.remember("User wants dark mode", session_id=session1)
await mem.end_session()

# Session 2 (next day)
session2 = await mem.start_session(parent_session_id=session1)
# Automatically gets context from session1 via bridging
results = await mem.recall("UI preferences")  # Includes session1 memories
```

---

## 📊 Monitoring & Debugging

```python
# Full statistics
stats = await mem.get_stats()
# {
#   "base": {"total_memories": 1500, "avg_importance": 0.65, ...},
#   "tiers": {"immediate": {"count": 5, "utilization": 0.5}, ...},
#   "sessions": {"total_sessions": 12, "active": 1, ...}
# }

# Tier-specific stats
tier_stats = await mem.get_tier_stats()
```

CLI also provides rich formatted output:
```bash
cognimem stats
cognimem tiers
cognimem session list
```

---

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=cognimem --cov-report=html
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Inspired by human memory research (Ebbinghaus, Atkinson-Shiffrin, Baddeley-Hitch)
- Built on ChromaDB, NetworkX, SentenceTransformers, and Pydantic
- Query planning inspired by LangChain and LlamaIndex architectures

---

**CogniMem v2.0** - Give your agents the memory they deserve. 🧠