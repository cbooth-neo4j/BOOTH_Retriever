"""BOOTH Retriever command-line interface.

Thin wrappers over ``BOOTHCurator`` and ``init_schema``. This scaffold only
wires up the command tree and ``--help`` text; individual subcommands raise
NotImplementedError until the underlying APIs are ported.
"""

from __future__ import annotations

import os

import typer

from . import __version__
from .schema import init_schema as _init_schema

app = typer.Typer(
    name="booth",
    help="BOOTH Retriever - curate approved Cypher queries for neo4j-graphrag.",
    no_args_is_help=True,
    add_completion=False,
)

curate_app = typer.Typer(
    name="curate",
    help="List, approve, reject and edit pending queries.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(curate_app)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"booth-retriever {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(  # noqa: B008 - Typer API
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """BOOTH command-line interface."""


@app.command()
def init(
    uri: str = typer.Option(
        None,
        "--uri",
        envvar="NEO4J_URI",
        help="Neo4j bolt URI. Defaults to NEO4J_URI env var.",
    ),
    user: str = typer.Option(
        "neo4j",
        "--user",
        envvar="NEO4J_USER",
        help="Neo4j username. Defaults to NEO4J_USER env var, then 'neo4j'.",
    ),
    password: str = typer.Option(
        None,
        "--password",
        envvar="NEO4J_PASSWORD",
        help="Neo4j password. Defaults to NEO4J_PASSWORD env var.",
    ),
    database: str = typer.Option(
        None,
        "--database",
        envvar="NEO4J_DATABASE",
        help="Neo4j database name for multi-database setups (optional).",
    ),
    embedding_dimensions: int = typer.Option(
        1536,
        "--dimensions",
        "-d",
        envvar="EMBEDDING_DIMENSIONS",
        help="Vector dimensions for the embedder you'll use. Must match.",
    ),
) -> None:
    """Bootstrap BOOTH's Neo4j schema (indexes, constraints). Idempotent."""
    from neo4j import GraphDatabase

    resolved_uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    if not password:
        typer.echo("Error: Neo4j password required (--password or NEO4J_PASSWORD).", err=True)
        raise typer.Exit(code=2)

    driver = GraphDatabase.driver(resolved_uri, auth=(user, password))
    try:
        result = _init_schema(
            driver,
            embedding_dimensions=embedding_dimensions,
            database=database,
        )
    finally:
        driver.close()

    if result.created:
        typer.echo(f"Created {len(result.created)} schema object(s):")
        for name in result.created:
            typer.echo(f"  + {name}")
    if result.already_existed:
        typer.echo(f"{len(result.already_existed)} schema object(s) already existed:")
        for name in result.already_existed:
            typer.echo(f"  . {name}")
    typer.echo(
        f"Vector index configured for {result.embedding_dimensions}-dim embeddings."
    )


@app.command()
def stats() -> None:
    """Print BOOTH stats: counts by status, cache hit rate."""
    raise NotImplementedError("booth stats: not yet wired up")


@app.command()
def feedback(
    query_id: str = typer.Argument(..., help="The query_id returned from retriever.query()"),
    helpful: bool = typer.Option(
        True,
        "--helpful/--not-helpful",
        help="Whether the answer was helpful.",
    ),
) -> None:
    """Submit user feedback for a completed query."""
    raise NotImplementedError("booth feedback: not yet wired up")


@curate_app.command("list")
def curate_list(
    status: str = typer.Option("pending_approval", help="Filter by status"),
    limit: int = typer.Option(50, help="Max rows to return"),
) -> None:
    """List pending queries awaiting curation."""
    raise NotImplementedError("booth curate list: not yet wired up")


@curate_app.command("show")
def curate_show(query_id: str = typer.Argument(..., help="Query UUID")) -> None:
    """Show full details of a pending query including attempts and responses."""
    raise NotImplementedError("booth curate show: not yet wired up")


@curate_app.command("approve")
def curate_approve(query_id: str = typer.Argument(..., help="Query UUID")) -> None:
    """Approve a query. Triggers RefinementAgent to create a FewShot."""
    raise NotImplementedError("booth curate approve: not yet wired up")


@curate_app.command("reject")
def curate_reject(
    query_id: str = typer.Argument(..., help="Query UUID"),
    reason: str = typer.Option("", help="Optional rejection reason"),
) -> None:
    """Reject a query."""
    raise NotImplementedError("booth curate reject: not yet wired up")


@curate_app.command("edit")
def curate_edit(
    fewshot_id: str = typer.Argument(..., help="FewShot UUID to edit"),
    cypher: str = typer.Option(..., help="New Cypher template, or @path/to/file.cypher"),
) -> None:
    """Edit an existing FewShot's Cypher template."""
    raise NotImplementedError("booth curate edit: not yet wired up")


if __name__ == "__main__":
    app()
