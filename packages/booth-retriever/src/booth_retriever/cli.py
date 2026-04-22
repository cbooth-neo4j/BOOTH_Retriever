"""BOOTH Retriever command-line interface.

Thin wrappers over ``init_schema`` and ``BOOTHCurator``. Every command
accepts Neo4j connection options (URI, user, password, database) either
via CLI flags or environment variables (``NEO4J_URI``, ``NEO4J_USER``,
``NEO4J_PASSWORD``, ``NEO4J_DATABASE``) and closes the driver on exit.

Quick reference:

    booth init                       # bootstrap schema
    booth curate list                # list pending queries
    booth curate show <query_id>     # show one query's detail
    booth curate approve <query_id> --cypher "RETURN 1" --params "a,b"
    booth curate approve <query_id> --cypher @path/to/template.cypher
    booth curate reject <query_id> --reason "off-topic"
    booth curate edit <query_id> --cypher "RETURN 2"
    booth feedback <query_id> --helpful
    booth stats
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import typer

from . import __version__
from .curator import ALL_STATUSES, BOOTHCurator, CuratorStats, QueryDetail
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


# ---------------------------------------------------------------------------
# Driver factory — overridable in tests so CliRunner can skip real Neo4j
# ---------------------------------------------------------------------------


def _default_driver_factory(uri: str, user: str, password: str):
    """Build a real neo4j.Driver. Factored out so tests can override it."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(uri, auth=(user, password))


_driver_factory: Callable = _default_driver_factory


def set_driver_factory(factory: Callable) -> None:
    """Override the driver factory (used by CLI tests)."""
    global _driver_factory
    _driver_factory = factory


def _build_driver_or_exit(
    uri: str | None,
    user: str,
    password: str | None,
):
    resolved_uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    if not password:
        typer.echo(
            "Error: Neo4j password required (--password or NEO4J_PASSWORD env var).",
            err=True,
        )
        raise typer.Exit(code=2)
    return _driver_factory(resolved_uri, user, password)


def _make_curator_or_exit(
    uri: str | None,
    user: str,
    password: str | None,
    database: str | None,
) -> tuple[BOOTHCurator, object]:
    """Return ``(curator, driver)``; caller must close the driver."""
    driver = _build_driver_or_exit(uri, user, password)
    curator = BOOTHCurator(driver=driver, database=database)
    return curator, driver


# ---------------------------------------------------------------------------
# Shared options — reduces duplication across commands
# ---------------------------------------------------------------------------


URI_OPT = typer.Option(
    None, "--uri", envvar="NEO4J_URI", help="Neo4j bolt URI. Defaults to NEO4J_URI env."
)
USER_OPT = typer.Option(
    "neo4j", "--user", envvar="NEO4J_USER", help="Neo4j username."
)
PASSWORD_OPT = typer.Option(
    None, "--password", envvar="NEO4J_PASSWORD", help="Neo4j password."
)
DATABASE_OPT = typer.Option(
    None,
    "--database",
    envvar="NEO4J_DATABASE",
    help="Neo4j database name for multi-database setups.",
)
JSON_OPT = typer.Option(
    False, "--json", help="Emit machine-readable JSON instead of human-readable text."
)


# ---------------------------------------------------------------------------
# Top-level callback (for --version)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# booth init
# ---------------------------------------------------------------------------


@app.command()
def init(
    uri: str = URI_OPT,
    user: str = USER_OPT,
    password: str = PASSWORD_OPT,
    database: str = DATABASE_OPT,
    embedding_dimensions: int = typer.Option(
        1536,
        "--dimensions",
        "-d",
        envvar="EMBEDDING_DIMENSIONS",
        help="Vector dimensions for the embedder you'll use.",
    ),
) -> None:
    """Bootstrap BOOTH's Neo4j schema (indexes, constraints). Idempotent."""
    driver = _build_driver_or_exit(uri, user, password)
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


# ---------------------------------------------------------------------------
# booth stats
# ---------------------------------------------------------------------------


@app.command()
def stats(
    uri: str = URI_OPT,
    user: str = USER_OPT,
    password: str = PASSWORD_OPT,
    database: str = DATABASE_OPT,
    json_out: bool = JSON_OPT,
) -> None:
    """Print counts of BOOTH queries by status."""
    curator, driver = _make_curator_or_exit(uri, user, password, database)
    try:
        s = curator.stats()
    finally:
        driver.close()
    _render_stats(s, json_out=json_out)


def _render_stats(s: CuratorStats, *, json_out: bool) -> None:
    if json_out:
        typer.echo(json.dumps({"total": s.total, "by_status": s.counts}, indent=2))
        return
    typer.echo(f"Total queries: {s.total}")
    for status, count in sorted(s.counts.items()):
        typer.echo(f"  {status:<20s} {count}")


# ---------------------------------------------------------------------------
# booth feedback
# ---------------------------------------------------------------------------


@app.command()
def feedback(
    query_id: str = typer.Argument(..., help="Query UUID returned from retriever.query()."),
    helpful: bool = typer.Option(
        True,
        "--helpful/--not-helpful",
        help="Whether the answer was helpful.",
    ),
    uri: str = URI_OPT,
    user: str = USER_OPT,
    password: str = PASSWORD_OPT,
    database: str = DATABASE_OPT,
) -> None:
    """Submit end-user feedback on a completed query."""
    curator, driver = _make_curator_or_exit(uri, user, password, database)
    try:
        curator.submit_feedback(query_id, helpful=helpful)
    except ValueError as e:
        driver.close()
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None
    driver.close()
    typer.echo(
        f"Recorded feedback for {query_id}: "
        f"{'helpful' if helpful else 'not_helpful'}."
    )


# ---------------------------------------------------------------------------
# booth curate list
# ---------------------------------------------------------------------------


@curate_app.command("list")
def curate_list(
    status: str = typer.Option(
        None,
        "--status",
        help=(
            "Filter by a single status. Omit to list all pending "
            "(pending_approval, declined, needs_review)."
        ),
    ),
    limit: int = typer.Option(50, "--limit", help="Max rows to return."),
    json_out: bool = JSON_OPT,
    uri: str = URI_OPT,
    user: str = USER_OPT,
    password: str = PASSWORD_OPT,
    database: str = DATABASE_OPT,
) -> None:
    """List pending queries awaiting curation."""
    if status is not None and status not in ALL_STATUSES:
        typer.echo(
            f"Error: unknown status {status!r}. Valid: {sorted(ALL_STATUSES)}",
            err=True,
        )
        raise typer.Exit(code=2)

    curator, driver = _make_curator_or_exit(uri, user, password, database)
    try:
        rows = (
            curator.list_by_status(status, limit=limit)
            if status
            else curator.list_pending(limit=limit)
        )
    finally:
        driver.close()

    if json_out:
        typer.echo(
            json.dumps(
                [
                    {
                        "query_id": r.query_id,
                        "query_text": r.query_text,
                        "status": r.status,
                        "risk_level": r.risk_level,
                        "timestamp": r.timestamp,
                        "user_feedback": r.user_feedback,
                        "has_fewshot": r.has_fewshot,
                    }
                    for r in rows
                ],
                indent=2,
            )
        )
        return

    if not rows:
        typer.echo("No queries found.")
        return
    typer.echo(f"Found {len(rows)} quer{'y' if len(rows) == 1 else 'ies'}:")
    for r in rows:
        marker = "*" if r.has_fewshot else " "
        typer.echo(
            f"  {marker} {r.query_id[:8]}... [{r.status:<18s}] "
            f"risk={r.risk_level or '-':<4s} {r.query_text[:80]}"
        )
    typer.echo("  (* = has FewShot)")


# ---------------------------------------------------------------------------
# booth curate show
# ---------------------------------------------------------------------------


@curate_app.command("show")
def curate_show(
    query_id: str = typer.Argument(..., help="Query UUID."),
    json_out: bool = JSON_OPT,
    uri: str = URI_OPT,
    user: str = USER_OPT,
    password: str = PASSWORD_OPT,
    database: str = DATABASE_OPT,
) -> None:
    """Show full detail for a single query, including any linked FewShot."""
    curator, driver = _make_curator_or_exit(uri, user, password, database)
    try:
        detail = curator.get(query_id)
    finally:
        driver.close()
    if detail is None:
        typer.echo(f"Error: no query with id {query_id!r}", err=True)
        raise typer.Exit(code=1)
    _render_detail(detail, json_out=json_out)


def _render_detail(d: QueryDetail, *, json_out: bool) -> None:
    if json_out:
        typer.echo(
            json.dumps(
                {
                    "query_id": d.query_id,
                    "query_text": d.query_text,
                    "status": d.status,
                    "risk_level": d.risk_level,
                    "timestamp": d.timestamp,
                    "user_feedback": d.user_feedback,
                    "rejection_reason": d.rejection_reason,
                    "fewshot_cypher": d.fewshot_cypher,
                    "fewshot_parameters": d.fewshot_parameters,
                },
                indent=2,
            )
        )
        return
    typer.echo(f"Query:       {d.query_id}")
    typer.echo(f"Text:        {d.query_text}")
    typer.echo(f"Status:      {d.status}")
    typer.echo(f"Risk level:  {d.risk_level or '-'}")
    typer.echo(f"Created:     {d.timestamp or '-'}")
    if d.user_feedback:
        typer.echo(f"Feedback:    {d.user_feedback}")
    if d.rejection_reason:
        typer.echo(f"Rejection:   {d.rejection_reason}")
    if d.fewshot_cypher:
        typer.echo("FewShot:")
        typer.echo(f"  parameters: {d.fewshot_parameters}")
        typer.echo(f"  cypher:     {d.fewshot_cypher}")
    else:
        typer.echo("FewShot:     (none)")


# ---------------------------------------------------------------------------
# booth curate approve / reject / edit
# ---------------------------------------------------------------------------


def _load_cypher(value: str) -> str:
    """Support ``--cypher @path/to/file.cypher`` for templates from disk."""
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        if not path.is_file():
            raise typer.BadParameter(f"File not found: {path}")
        return path.read_text(encoding="utf-8")
    return value


def _parse_params(value: str | None) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


@curate_app.command("approve")
def curate_approve(
    query_id: str = typer.Argument(..., help="Query UUID."),
    cypher: str = typer.Option(
        ...,
        "--cypher",
        "-c",
        help="Cypher template to approve, or @path/to/file.cypher to read from disk.",
    ),
    params: str = typer.Option(
        None,
        "--params",
        help="Comma-separated parameter names referenced in the template (e.g. 'name,role').",
    ),
    uri: str = URI_OPT,
    user: str = USER_OPT,
    password: str = PASSWORD_OPT,
    database: str = DATABASE_OPT,
) -> None:
    """Approve a query and attach (or replace) a FewShot cypher template."""
    template = _load_cypher(cypher)
    param_list = _parse_params(params)

    curator, driver = _make_curator_or_exit(uri, user, password, database)
    try:
        result = curator.approve(
            query_id,
            cypher_template=template,
            parameters=param_list,
        )
    except ValueError as e:
        driver.close()
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None
    driver.close()

    verb = "created" if result.fewshot_was_new else "updated"
    typer.echo(f"Approved {query_id}; FewShot {verb} ({result.fewshot_id}).")
    if param_list:
        typer.echo(f"  parameters: {param_list}")


@curate_app.command("reject")
def curate_reject(
    query_id: str = typer.Argument(..., help="Query UUID."),
    reason: str = typer.Option(None, "--reason", "-r", help="Optional rejection reason."),
    uri: str = URI_OPT,
    user: str = USER_OPT,
    password: str = PASSWORD_OPT,
    database: str = DATABASE_OPT,
) -> None:
    """Reject a query with an optional reason."""
    curator, driver = _make_curator_or_exit(uri, user, password, database)
    try:
        curator.reject(query_id, reason=reason)
    except ValueError as e:
        driver.close()
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None
    driver.close()
    msg = f"Rejected {query_id}."
    if reason:
        msg += f" Reason: {reason}"
    typer.echo(msg)


@curate_app.command("edit")
def curate_edit(
    query_id: str = typer.Argument(
        ..., help="Query UUID whose linked FewShot you want to edit."
    ),
    cypher: str = typer.Option(
        ...,
        "--cypher",
        "-c",
        help="New Cypher template, or @path/to/file.cypher.",
    ),
    params: str = typer.Option(
        None,
        "--params",
        help="Comma-separated parameter names.",
    ),
    uri: str = URI_OPT,
    user: str = USER_OPT,
    password: str = PASSWORD_OPT,
    database: str = DATABASE_OPT,
) -> None:
    """Edit the Cypher template on an already-approved query's FewShot."""
    template = _load_cypher(cypher)
    param_list = _parse_params(params)

    curator, driver = _make_curator_or_exit(uri, user, password, database)
    try:
        curator.edit_fewshot(
            query_id,
            cypher_template=template,
            parameters=param_list,
        )
    except ValueError as e:
        driver.close()
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None
    driver.close()
    typer.echo(f"Updated FewShot for {query_id}.")


if __name__ == "__main__":
    app()
