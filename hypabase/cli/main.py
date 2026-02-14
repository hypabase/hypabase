"""Hypabase CLI — command-line interface for hypergraph operations."""

from __future__ import annotations

import json
from pathlib import Path

import click

from hypabase.client import Hypabase

DEFAULT_DB = "hypabase.db"


def _get_client(db: str, database: str = "default") -> Hypabase:
    hb = Hypabase(db)
    if database != "default":
        return hb.database(database)
    return hb


@click.group()
@click.option("--db", default=DEFAULT_DB, help="Path to the database file.")
@click.option("--database", default="default", help="Namespace (default: 'default').")
@click.pass_context
def cli(ctx: click.Context, db: str, database: str) -> None:
    """Hypabase CLI — manage hypergraphs from the command line."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db
    ctx.obj["database"] = database


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize a new Hypabase project in the current directory."""
    db_path = ctx.obj["db"]
    if Path(db_path).exists():
        click.echo(f"Database already exists at {db_path}")
        return
    hb = Hypabase(db_path)
    hb.close()
    click.echo(f"Initialized Hypabase database at {db_path}")


@cli.command()
@click.argument("id")
@click.option("--type", "node_type", default="unknown", help="Node type.")
@click.option("--props", default=None, help="JSON properties.")
@click.pass_context
def node(ctx: click.Context, id: str, node_type: str, props: str | None) -> None:
    """Create or update a node."""
    hb = _get_client(ctx.obj["db"], ctx.obj["database"])
    properties = json.loads(props) if props else {}
    result = hb.node(id, type=node_type, **properties)
    hb.close()
    click.echo(f"Node: {result.id} (type={result.type})")


@cli.command()
@click.argument("node_ids", nargs=-1, required=True)
@click.option("--type", "edge_type", required=True, help="Edge type.")
@click.option("--source", default=None, help="Provenance source.")
@click.option("--confidence", default=None, type=float, help="Confidence score.")
@click.option("--props", default=None, help="JSON properties.")
@click.pass_context
def edge(
    ctx: click.Context,
    node_ids: tuple[str, ...],
    edge_type: str,
    source: str | None,
    confidence: float | None,
    props: str | None,
) -> None:
    """Create a hyperedge connecting nodes."""
    hb = _get_client(ctx.obj["db"], ctx.obj["database"])
    kwargs: dict = {"type": edge_type}
    if source:
        kwargs["source"] = source
    if confidence is not None:
        kwargs["confidence"] = confidence
    if props:
        kwargs["properties"] = json.loads(props)
    result = hb.edge(list(node_ids), **kwargs)
    hb.close()
    click.echo(f"Edge: {result.id} (type={result.type}, nodes={result.node_ids})")


@cli.command()
@click.option("--containing", multiple=True, help="Filter by node IDs.")
@click.option("--type", "edge_type", default=None, help="Filter by edge type.")
@click.option("--match-all", is_flag=True, help="Require all containing nodes.")
@click.pass_context
def query(
    ctx: click.Context, containing: tuple[str, ...], edge_type: str | None, match_all: bool
) -> None:
    """Query edges in the hypergraph."""
    hb = _get_client(ctx.obj["db"], ctx.obj["database"])
    results = hb.edges(
        containing=list(containing) if containing else None,
        type=edge_type,
        match_all=match_all,
    )
    hb.close()
    if not results:
        click.echo("No edges found.")
        return
    for e in results:
        click.echo(
            f"  {e.id}  type={e.type}  nodes={e.node_ids}"
            f"  source={e.source}  confidence={e.confidence}"
        )


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show database statistics."""
    hb = _get_client(ctx.obj["db"], ctx.obj["database"])
    s = hb.stats()
    hb.close()
    click.echo(f"Nodes: {s.node_count}  Edges: {s.edge_count}")
    if s.nodes_by_type:
        click.echo("Nodes by type:")
        for t, c in s.nodes_by_type.items():
            click.echo(f"  {t}: {c}")
    if s.edges_by_type:
        click.echo("Edges by type:")
        for t, c in s.edges_by_type.items():
            click.echo(f"  {t}: {c}")


@cli.command()
@click.pass_context
def validate(ctx: click.Context) -> None:
    """Validate internal consistency of the hypergraph."""
    hb = _get_client(ctx.obj["db"], ctx.obj["database"])
    result = hb.validate()
    hb.close()
    if result.valid:
        click.echo("Hypergraph is valid.")
    else:
        click.echo("Validation errors:")
        for err in result.errors:
            click.echo(f"  ERROR: {err}")
    for warn in result.warnings:
        click.echo(f"  WARNING: {warn}")


@cli.command("export-hif")
@click.argument("output", type=click.Path())
@click.pass_context
def export_hif(ctx: click.Context, output: str) -> None:
    """Export the hypergraph to HIF (JSON) format."""
    hb = _get_client(ctx.obj["db"], ctx.obj["database"])
    hif_data = hb.to_hif()
    hb.close()
    Path(output).write_text(json.dumps(hif_data, indent=2))
    click.echo(f"Exported HIF to {output}")


@cli.command("import-hif")
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def import_hif(ctx: click.Context, input_file: str) -> None:
    """Import a hypergraph from HIF (JSON) format."""
    hif_data = json.loads(Path(input_file).read_text())
    db_path = ctx.obj["db"]
    database = ctx.obj["database"]
    hb_imported = Hypabase.from_hif(hif_data)

    # Persist to the target database
    with Hypabase(db_path) as hb:
        target = hb.database(database) if database != "default" else hb
        # Copy all nodes and edges from imported data
        for node in hb_imported.nodes():
            target.node(node.id, type=node.type, **node.properties)
        for edge in hb_imported.edges():
            target.edge(
                edge.node_ids,
                type=edge.type,
                source=edge.source,
                confidence=edge.confidence,
                properties=edge.properties,
            )
    click.echo(f"Imported HIF from {input_file} into {db_path} (database={database})")


@cli.command()
@click.option("--db", default=None, help="Database path (overrides HYPABASE_DB_PATH).")
def mcp(db: str | None) -> None:
    """Start the MCP server for AI agent integration."""
    import os

    if db:
        os.environ["HYPABASE_DB_PATH"] = db
    from hypabase.mcp.server import run_server

    run_server()


if __name__ == "__main__":
    cli()
