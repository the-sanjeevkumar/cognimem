#!/usr/bin/env python
"""
CogniMem v2.0 - Main Entry Point
Demonstration of the advanced cognitive memory system.
"""

import asyncio
from cognimem import create_cognimem
from cognimem.core.hierarchical import MemoryTier


async def main():
    print("🧠 CogniMem v2.0 - Advanced Cognitive Memory System")
    print("=" * 60)
    
    # Create and initialize memory system
    mem = await create_cognimem(agent_id="demo_agent")
    async with mem:
        
        # 1. Store memories with different importance levels
        print("\n📝 Storing memories...")
        
        await mem.remember(
            content="The user's name is Alice and she prefers Python for backend development.",
            importance=0.9,
            tags=["user_profile", "python", "preferences"]
        )
        
        await mem.remember(
            content="We discussed deploying the application using Docker and Kubernetes yesterday.",
            importance=0.7,
            tags=["project", "deployment", "docker", "kubernetes"]
        )
        
        await mem.remember(
            content="The weather was rainy during our meeting.",
            importance=0.2,
            tags=["weather", "context"]
        )
        
        await mem.remember(
            content="Alice decided to use PostgreSQL as the primary database for the project.",
            importance=0.8,
            tags=["decision", "database", "postgresql"]
        )
        
        print("✅ Memories stored")
        
        # 2. Start a session
        print("\n📋 Starting session...")
        session_id = await mem.start_session(tags=["demo", "meeting"])
        print(f"Session started: {session_id}")
        
        # 3. Recall with different strategies
        print("\n🔍 Recalling memories...")
        
        # Simple recall
        print("\n--- Simple Recall ---")
        result = await mem.recall_simple("What programming language does Alice prefer?")
        print(result)
        
        # Advanced recall with query planner
        print("\n--- Advanced Recall (with query planner) ---")
        results = await mem.recall(
            query="What database did Alice choose and why?",
            top_k=3,
            use_query_planner=True
        )
        for r in results:
            print(f"  [{r.final_score:.2f}] {r.record.content}")
        
        # 4. Tier-specific recall
        print("\n--- Tier-specific Recall (long_term only) ---")
        results = await mem.recall(
            query="deployment",
            top_k=3,
            tiers=[MemoryTier.LONG_TERM]
        )
        for r in results:
            print(f"  [{r.final_score:.2f}] {r.record.content}")
        
        # 5. Proactive memory
        print("\n🎯 Proactive Memory Demo...")
        results = await mem.on_context_change(
            context={"current_goal": "Deploy the application", "current_topic": "database selection"},
            max_results=2
        )
        print("Context-aware prefetch results:")
        for r in results:
            print(f"  [{r.final_score:.2f}] {r.record.content}")
        
        # 6. Statistics
        print("\n📊 Statistics...")
        stats = await mem.get_stats()
        print(f"Total memories: {stats['base']['total_memories']}")
        print(f"Avg importance: {stats['base']['avg_importance']:.2f}")
        print(f"By tier: {stats['base']['by_tier']}")
        
        tier_stats = await mem.get_tier_stats()
        print("\nTier details:")
        for tier_name, info in tier_stats.get('tiers', {}).items():
            print(f"  {tier_name}: {info['count']}/{info['capacity']} ({info['utilization']:.1%})")
        
        # 7. Session management
        print("\n📋 Session Management...")
        sessions = await mem.list_sessions()
        print(f"Active sessions: {len(sessions)}")
        for s in sessions:
            print(f"  {s['session_id'][:20]}... [{s['status']}] memories: {s['memory_count']}")
        
        # End session
        await mem.end_session()
        print("Session ended with summary")
        
        # 8. Shared memory (multi-agent)
        print("\n🤝 Multi-Agent Shared Memory...")
        space = await mem.create_shared_space("Project Team", namespace="team")
        print(f"Created shared space: {space['space_id']}")
        
        await mem.remember_shared(
            space_id=space['space_id'],
            content="Team decision: Use Redis for caching layer",
            importance=0.8,
            tags=["team_decision", "redis", "caching"]
        )
        
        shared_results = await mem.recall_shared(
            space_id=space['space_id'],
            query="caching decision",
            top_k=2
        )
        print("Shared memory recall:")
        for r in shared_results:
            print(f"  [{r.final_score:.2f}] {r.record.content}")
        
        # 9. Memory compression
        print("\n🗜️ Memory Compression...")
        compressed = await mem.auto_compress_old_memories(age_threshold_days=0, level="medium")
        print(f"Compressed {compressed} old memories")
        
        # 10. Rebalance and cleanup
        print("\n🔧 Maintenance...")
        moved = await mem.rebalance_tiers()
        print(f"Rebalanced: {moved}")
        
        cleaned = await mem.cleanup_expired()
        print(f"Cleaned up: {cleaned} expired memories")
        
        print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())