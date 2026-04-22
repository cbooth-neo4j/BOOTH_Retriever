"""BOOTH Retriever command-line interface.

Thin wrappers over ``BOOTHCurator`` and ``init_schema``. This scaffold only
wires up the command tree and ``--help`` text; individual subcommands raise
NotImplementedError until the underlying APIs are ported.
"""

from __future__ import annotations

import typer

from . import __version__

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
def init() -> None:
    """Bootstrap BOOTH's Neo4j schema (indexes, constraints). Idempotent."""
    raise NotImplementedError("booth init: not yet wired up")


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
