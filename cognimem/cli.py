#!/usr/bin/env python
"""
CogniMem v2.0 - Command Line Interface
Memory management and inspection tools.
"""

import asyncio
import json
import sys
from typing import Optional, List, Dict
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from cognimem import create_cognimem, get_settings
from cognimem.config import CogniMemSettings
from cognimem.core.hierarchical import MemoryTier


console = Console()


@click.group()
@click.option('--agent-id', default='default_agent', help='Agent ID')
@click.option('--config', default='config.yaml', help='Config file path')
@click.pass_context
def cli(ctx, agent_id: str, config: str):
    """CogniMem v2.0 - Advanced Cognitive Memory for AI Agents"""
    ctx.ensure_object(dict)
    ctx.obj['agent_id'] = agent_id
    ctx.obj['config'] = config


@cli.command()
@click.argument('content')
@click.option('--importance', '-i', default=0.5, type=float, help='Importance score (0-1)')
@click.option('--tags', '-t', multiple=True, help='Tags for the memory')
@click.option('--tier', type=click.Choice(['immediate', 'short_term', 'long_term', 'archival']), help='Memory tier')
@click.pass_context
def remember(ctx, content: str, importance: float, tags: tuple, tier: str):
    """Store a new memory."""
    agent_id = ctx.obj['agent_id']
    
    async def _remember():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            memory_id = await mem.remember(
                content=content,
                importance=importance,
                tags=list(tags),
                tier=MemoryTier(tier) if tier else None
            )
            console.print(f"[green]✓[/green] Memory stored with ID: {memory_id}")
            return memory_id
    
    asyncio.run(_remember())


@cli.command()
@click.argument('query')
@click.option('--top-k', '-k', default=5, type=int, help='Number of results')
@click.option('--tier', type=click.Choice(['immediate', 'short_term', 'long_term', 'archival']), multiple=True, help='Search specific tiers')
@click.option('--min-importance', default=0.0, type=float, help='Minimum importance')
@click.option('--simple', '-s', is_flag=True, help='Simple text output')
@click.pass_context
def recall(ctx, query: str, top_k: int, tier: tuple, min_importance: float, simple: bool):
    """Recall memories matching a query."""
    agent_id = ctx.obj['agent_id']
    
    async def _recall():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            tiers = [MemoryTier(t) for t in tier] if tier else None
            results = await mem.recall(
                query=query,
                top_k=top_k,
                tiers=tiers,
                min_importance=min_importance
            )
            
            if simple:
                if not results:
                    console.print("No relevant memories found.")
                    return
                for r in results:
                    console.print(f"[{r.record.timestamp.strftime('%Y-%m-%d %H:%M')}] (Score: {r.final_score:.2f}): {r.record.content}")
            else:
                _display_results(results)
    
    asyncio.run(_recall())


@cli.command()
@click.pass_context
def stats(ctx):
    """Show memory statistics."""
    agent_id = ctx.obj['agent_id']
    
    async def _stats():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            stats = await mem.get_stats()
            _display_stats(stats)
    
    asyncio.run(_stats())


@cli.command()
@click.pass_context
def tiers(ctx):
    """Show tier statistics."""
    agent_id = ctx.obj['agent_id']
    
    async def _tiers():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            tier_stats = await mem.get_tier_stats()
            _display_tier_stats(tier_stats)
    
    asyncio.run(_tiers())


@cli.command()
@click.option('--max-age-days', default=30, type=int, help='Maximum age in days')
@click.option('--level', type=click.Choice(['light', 'medium', 'aggressive']), default='aggressive', help='Compression level')
@click.pass_context
def compress(ctx, max_age_days: int, level: str):
    """Compress old memories."""
    agent_id = ctx.obj['agent_id']
    
    async def _compress():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            count = await mem.auto_compress_old_memories(max_age_days, level)
            console.print(f"[green]✓[/green] Compressed {count} memories")
    
    asyncio.run(_compress())


@cli.command()
@click.pass_context
def rebalance(ctx):
    """Rebalance memories across tiers."""
    agent_id = ctx.obj['agent_id']
    
    async def _rebalance():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            moved = await mem.rebalance_tiers()
            console.print(f"[green]✓[/green] Rebalanced tiers: {moved}")
    
    asyncio.run(_rebalance())


@cli.command()
@click.pass_context
def cleanup(ctx):
    """Clean up expired memories."""
    agent_id = ctx.obj['agent_id']
    
    async def _cleanup():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            count = await mem.cleanup_expired()
            console.print(f"[green]✓[/green] Cleaned up {count} expired memories")
    
    asyncio.run(_cleanup())


# Session commands
@cli.group()
def session():
    """Session management commands."""
    pass


@session.command('start')
@click.option('--parent', help='Parent session ID')
@click.option('--tags', '-t', multiple=True, help='Session tags')
@click.pass_context
def session_start(ctx, parent: str, tags: tuple):
    """Start a new session."""
    agent_id = ctx.obj['agent_id']
    
    async def _start():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            session_id = await mem.start_session(parent_session_id=parent, tags=list(tags))
            console.print(f"[green]✓[/green] Started session: {session_id}")
    
    asyncio.run(_start())


@session.command('end')
@click.option('--no-summary', is_flag=True, help='Skip summary creation')
@click.pass_context
def session_end(ctx, no_summary: bool):
    """End the current session."""
    agent_id = ctx.obj['agent_id']
    
    async def _end():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            result = await mem.end_session(create_summary=not no_summary)
            if result:
                console.print("[green]✓[/green] Session ended")
            else:
                console.print("[yellow]![/yellow] No active session")
    
    asyncio.run(_end())


@session.command('list')
@click.option('--status', type=click.Choice(['active', 'paused', 'ended', 'archived']), help='Filter by status')
@click.option('--limit', default=20, type=int, help='Maximum sessions to show')
@click.pass_context
def session_list(ctx, status: str, limit: int):
    """List sessions."""
    agent_id = ctx.obj['agent_id']
    
    async def _list():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            sessions = await mem.list_sessions(status=status, limit=limit)
            _display_sessions(sessions)
    
    asyncio.run(_list())


@session.command('info')
@click.argument('session_id')
@click.pass_context
def session_info(ctx, session_id: str):
    """Show detailed session information."""
    agent_id = ctx.obj['agent_id']
    
    async def _info():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            session = await mem.session_manager.get_session(session_id)
            if not session:
                console.print("[red]✗[/red] Session not found")
                return
            
            _display_session_detail(session)
    
    asyncio.run(_info())


# Multi-agent commands
@cli.group()
def shared():
    """Shared memory space commands."""
    pass


@shared.command('create')
@click.argument('name')
@click.option('--namespace', type=click.Choice(['personal', 'team', 'org', 'public']), default='team', help='Namespace type')
@click.pass_context
def shared_create(ctx, name: str, namespace: str):
    """Create a shared memory space."""
    agent_id = ctx.obj['agent_id']
    
    async def _create():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            space = await mem.create_shared_space(name=name, namespace=namespace)
            console.print(f"[green]✓[/green] Created shared space: {space['space_id']} ({space['name']})")
    
    asyncio.run(_create())


@shared.command('remember')
@click.argument('space_id')
@click.argument('content')
@click.option('--importance', '-i', default=0.5, type=float, help='Importance score')
@click.option('--tags', '-t', multiple=True, help='Tags')
@click.pass_context
def shared_remember(ctx, space_id: str, content: str, importance: float, tags: tuple):
    """Store a memory in a shared space."""
    agent_id = ctx.obj['agent_id']
    
    async def _remember():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            memory_id = await mem.remember_shared(
                space_id=space_id,
                content=content,
                importance=importance,
                tags=list(tags)
            )
            if memory_id:
                console.print(f"[green]✓[/green] Memory stored: {memory_id}")
            else:
                console.print("[red]✗[/red] Permission denied or space not found")
    
    asyncio.run(_remember())


@shared.command('recall')
@click.argument('space_id')
@click.argument('query')
@click.option('--top-k', '-k', default=5, type=int, help='Number of results')
@click.pass_context
def shared_recall(ctx, space_id: str, query: str, top_k: int):
    """Recall memories from a shared space."""
    agent_id = ctx.obj['agent_id']
    
    async def _recall():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            results = await mem.recall_shared(space_id=space_id, query=query, top_k=top_k)
            _display_results(results)
    
    asyncio.run(_recall())


@shared.command('list')
@click.pass_context
def shared_list(ctx):
    """List accessible shared spaces."""
    agent_id = ctx.obj['agent_id']
    
    async def _list():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            spaces = await mem.multiagent.list_spaces(agent_id)
            _display_shared_spaces(spaces)
    
    asyncio.run(_list())


# Proactive commands
@cli.group()
def proactive():
    """Proactive memory commands."""
    pass


@proactive.command('context')
@click.argument('key')
@click.argument('value')
@click.option('--max-results', default=3, type=int, help='Maximum results')
@click.pass_context
def proactive_context(ctx, key: str, value: str, max_results: int):
    """Trigger proactive retrieval on context change."""
    agent_id = ctx.obj['agent_id']
    
    async def _context():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            results = await mem.on_context_change({key: value}, max_results=max_results)
            _display_results(results)
    
    asyncio.run(_context())


@proactive.command('entity')
@click.argument('entity')
@click.option('--max-results', default=3, type=int, help='Maximum results')
@click.pass_context
def proactive_entity(ctx, entity: str, max_results: int):
    """Trigger proactive retrieval on entity mention."""
    agent_id = ctx.obj['agent_id']
    
    async def _entity():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            results = await mem.on_entity_mention(entity, {}, max_results=max_results)
            _display_results(results)
    
    asyncio.run(_entity())


@proactive.command('decision')
@click.argument('decision')
@click.option('--max-results', default=5, type=int, help='Maximum results')
@click.pass_context
def proactive_decision(ctx, decision: str, max_results: int):
    """Trigger proactive retrieval for pending decision."""
    agent_id = ctx.obj['agent_id']
    
    async def _decision():
        mem = await create_cognimem(agent_id=agent_id)
        async with mem:
            results = await mem.on_decision_pending(decision, {}, max_results=max_results)
            _display_results(results)
    
    asyncio.run(_decision())


# Config commands
@cli.group()
def config():
    """Configuration commands."""
    pass


@config.command('show')
@click.pass_context
def config_show(ctx):
    """Show current configuration."""
    settings = get_settings()
    console.print(Syntax(settings.model_dump_json(indent=2), "json"))


@config.command('validate')
@click.pass_context
def config_validate(ctx):
    """Validate configuration."""
    try:
        settings = get_settings()
        console.print("[green]✓[/green] Configuration is valid")
        console.print(f"Agent ID: {settings.agent_id}")
        console.print(f"Embedding model: {settings.embedding.model}")
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")


# Utility functions

def _display_results(results: List):
    """Display search results in a table."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    table = Table(title="Memory Search Results")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("Tier", style="magenta")
    table.add_column("Time", style="dim")
    table.add_column("Content", style="white")
    table.add_column("Tags", style="green")
    
    for r in results:
        table.add_row(
            f"{r.final_score:.3f}",
            r.record.tier,
            r.record.timestamp.strftime("%Y-%m-%d %H:%M"),
            r.record.content[:80] + ("..." if len(r.record.content) > 80 else ""),
            ", ".join(r.record.tags[:3]) + ("..." if len(r.record.tags) > 3 else "")
        )
    
    console.print(table)


def _display_stats(stats: Dict):
    """Display memory statistics."""
    console.print(Panel.fit(
        f"[bold]Agent:[/bold] {stats.get('agent_id', 'N/A')}\n"
        f"[bold]Total Memories:[/bold] {stats.get('base', {}).get('total_memories', 0)}\n"
        f"[bold]Avg Importance:[/bold] {stats.get('base', {}).get('avg_importance', 0):.2f}\n"
        f"[bold]By Type:[/bold] {stats.get('base', {}).get('by_type', {})}\n"
        f"[bold]By Tier:[/bold] {stats.get('base', {}).get('by_tier', {})}",
        title="Memory Statistics",
        border_style="blue"
    ))


def _display_tier_stats(tier_stats: Dict):
    """Display tier statistics."""
    tiers = tier_stats.get('tiers', {})
    
    table = Table(title="Tier Statistics")
    table.add_column("Tier", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Capacity", justify="right")
    table.add_column("Utilization", justify="right")
    table.add_column("TTL (hrs)", justify="right")
    table.add_column("Min Importance", justify="right")
    
    for tier_name, info in tiers.items():
        util = info.get('utilization', 0)
        util_color = "green" if util < 0.5 else "yellow" if util < 0.8 else "red"
        table.add_row(
            tier_name,
            str(info.get('count', 0)),
            str(info.get('capacity', 0)),
            f"[{util_color}]{util:.1%}[/{util_color}]",
            str(info.get('ttl_hours', 0)),
            str(info.get('min_importance', 0))
        )
    
    console.print(table)


def _display_sessions(sessions: List[Dict]):
    """Display sessions."""
    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return
    
    table = Table(title="Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Created", style="dim")
    table.add_column("Updated", style="dim")
    table.add_column("Memories", justify="right")
    table.add_column("Tags", style="green")
    
    for s in sessions:
        status_color = {
            'active': 'green',
            'paused': 'yellow',
            'ended': 'blue',
            'archived': 'dim'
        }.get(s['status'], 'white')
        
        table.add_row(
            s['session_id'],
            f"[{status_color}]{s['status']}[/{status_color}]",
            s['created_at'][:19],
            s['updated_at'][:19],
            str(s['memory_count']),
            ", ".join(s['tags'][:3])
        )
    
    console.print(table)


def _display_session_detail(session):
    """Display detailed session info."""
    console.print(Panel(
        f"[bold]Session ID:[/bold] {session.session_id}\n"
        f"[bold]Agent:[/bold] {session.agent_id}\n"
        f"[bold]Status:[/bold] {session.status.value}\n"
        f"[bold]Created:[/bold] {session.created_at}\n"
        f"[bold]Updated:[/bold] {session.updated_at}\n"
        f"[bold]Ended:[/bold] {session.ended_at or 'N/A'}\n"
        f"[bold]Memory Count:[/bold] {session.memory_count}\n"
        f"[bold]Tags:[/bold] {', '.join(session.tags) or 'None'}\n"
        f"[bold]Parent:[/bold] {session.parent_session_id or 'None'}\n"
        f"[bold]Children:[/bold] {', '.join(session.child_session_ids) or 'None'}\n"
        f"[bold]Summary:[/bold]\n{session.summary or 'No summary'}",
        title=f"Session: {session.session_id}",
        border_style="blue"
    ))


def _display_shared_spaces(spaces):
    """Display shared memory spaces."""
    if not spaces:
        console.print("[yellow]No shared spaces found.[/yellow]")
        return
    
    table = Table(title="Shared Memory Spaces")
    table.add_column("Space ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Namespace", style="magenta")
    table.add_column("Owner", style="dim")
    table.add_column("Members", justify="right")
    
    for s in spaces:
        table.add_row(
            s.space_id,
            s.name,
            s.namespace.value,
            s.owner_id,
            str(len(s.members))
        )
    
    console.print(table)


if __name__ == '__main__':
    cli()