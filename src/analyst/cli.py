"""Command line entry point."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from analyst.adapters.fake_model import FakePatternModel
from analyst.adapters.loader import UnsupportedFormat, discover
from analyst.app import update
from analyst.app.runs import run_id_for
from analyst.domain.delta import RegisterDelta
from analyst.domain.models import Decision, ProposalKind
from analyst.graph.build import compile_graph
from analyst.graph.serde import proposal_from, proposal_to, register_from
from analyst.settings import MissingCredential, Settings

app = typer.Typer(
    add_completion=False,
    help="Read a pile of related documents, build one cited register, check it "
    "against your rules, and keep it current as new documents arrive.",
)
console = Console()

KIND_STYLES: dict[ProposalKind, str] = {
    ProposalKind.ADD_OBLIGATION: "green",
    ProposalKind.SUPERSEDE_OBLIGATION: "blue",
    ProposalKind.CONFLICT: "yellow",
    ProposalKind.FINDING: "magenta",
    ProposalKind.QUARANTINE: "red",
}


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def _report_arrivals(corpus: Path, state: dict[str, Any], run_id: str) -> None:
    """Name documents sitting in the folder that this run never read.

    A finished run reports the corpus as it stood, so a file added since is
    invisible unless it is pointed at.
    """
    unread = [path for path in discover(corpus) if path.stem not in (state.get("sources") or {})]
    if not unread:
        return

    console.print(
        f"[yellow]{len(unread)} document(s) here are not in this run:[/yellow] "
        f"{', '.join(path.name for path in unread)}"
    )
    console.print(f"[dim]add one with: analyst ingest {run_id} {unread[0]}[/dim]")


def _graph(database: Path) -> tuple[Any, sqlite3.Connection, str]:
    """The graph, built against whichever model the environment selects.

    Defaults to the pattern-backed stand-in, so this runs with no key.
    """
    settings = Settings()
    try:
        model = settings.build_model()
    except MissingCredential as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    graph, connection = compile_graph(model, database)
    return graph, connection, str(settings.llm_provider)


@app.command()
def run(
    corpus: Annotated[Path, typer.Argument(help="Folder of documents to read.")] = Path("corpus"),
    rules: Annotated[Path, typer.Option(help="Rules file to check against.")] = Path(
        "corpus/rules/vendor-billing-checklist.yaml"
    ),
    run_id: Annotated[str | None, typer.Option(help="Resume a run by its id.")] = None,
    database: Annotated[Path, typer.Option(help="Where run state is kept.")] = Path("runs.db"),
    yes: Annotated[
        bool, typer.Option("--yes", help="Approve every register row without asking.")
    ] = False,
    new: Annotated[
        bool, typer.Option("--new", help="Read the corpus again as a separate run.")
    ] = False,
) -> None:
    """Run the pipeline up to the human gate."""
    if not corpus.exists():
        console.print(f"[red]No such folder:[/red] {corpus}")
        raise typer.Exit(code=2)

    identifier = run_id or run_id_for(corpus, rules, fresh=new)
    graph, connection, provider = _graph(database)

    try:
        console.print(f"[dim]run {identifier}  ·  {corpus}  ·  {provider}[/dim]\n")

        existing = graph.get_state(_config(identifier))

        # A finished run is reported rather than repeated. Reading the same
        # documents again would cost the same again and would throw away the
        # decisions a person already made on them.
        if existing.values and not existing.next:
            _show_stages(existing.values)
            _show_register(existing.values)
            _show_outcome(existing.values)
            _report_arrivals(corpus, existing.values, identifier)
            console.print("[dim]already decided and committed. --new reads it again.[/dim]")
            return

        # A run that stopped at the gate is continued, not started again.
        # Passing input a second time would re-enter at the first stage and pay
        # for work the checkpoint already holds.
        resuming = bool(existing.values) and bool(existing.next)
        if resuming:
            console.print(f"[dim]resuming at {existing.next[0]}[/dim]\n")

        state = (
            existing.values
            if resuming
            else graph.invoke(
                {
                    "run_id": identifier,
                    "corpus_dir": str(corpus),
                    "rules_path": str(rules),
                },
                config=_config(identifier),
            )
        )

        _show_stages(state)
        _show_register(state)

        proposals = [proposal_from(p) for p in state.get("proposals") or []]
        if not proposals:
            console.print("[yellow]Nothing to decide.[/yellow]")
            return

        decided = _review(proposals, approve_all=yes)
        if decided is None:
            console.print(
                f"\n[yellow]Stopped before deciding everything. Nothing was written.[/yellow]\n"
                f"[dim]Resume with: analyst run {corpus}[/dim]"
            )
            return

        # The graph is paused before commit. Supplying decisions and resuming is
        # what lets anything reach the register.
        graph.update_state(_config(identifier), {"proposals": [proposal_to(p) for p in decided]})
        final = graph.invoke(None, config=_config(identifier))

        _show_outcome(final)
        console.print(f"[dim]run id {identifier}[/dim]")
    finally:
        connection.close()


@app.command()
def ingest(
    run_id: Annotated[str, typer.Argument(help="The run to add the document to.")],
    document: Annotated[Path, typer.Argument(help="The document that arrived.")],
    database: Annotated[Path, typer.Option()] = Path("runs.db"),
    yes: Annotated[
        bool, typer.Option("--yes", help="Approve every register row without asking.")
    ] = False,
) -> None:
    """Add a document to a finished run and show what it changed.

    Not a re-run. Sources already read are not read again, facts already
    grounded are not extracted again, and rules already checked are not checked
    again. The report names what moved and proves what did not.
    """
    if not document.exists():
        console.print(f"[red]No such file:[/red] {document}")
        raise typer.Exit(code=2)

    graph, connection, _ = _graph(database)
    try:
        try:
            state, delta = update.ingest(graph, run_id, document)
        except (update.NotUpdatable, UnsupportedFormat) as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=1) from error

        console.print(f"[dim]{document.name} into run {run_id}[/dim]\n")
        _show_delta(delta)

        proposals = [proposal_from(p) for p in state.get("proposals") or []]
        if proposals:
            decided = _review(proposals, approve_all=yes)
            if decided is None:
                console.print("\n[yellow]Stopped. Nothing was written.[/yellow]")
                return
            graph.update_state(_config(run_id), {"proposals": [proposal_to(p) for p in decided]})
            _show_outcome(graph.invoke(None, config=_config(run_id)))
    finally:
        connection.close()


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="The run to inspect.")],
    database: Annotated[Path, typer.Option()] = Path("runs.db"),
    as_json: Annotated[bool, typer.Option("--json", help="Print the register as JSON.")] = False,
) -> None:
    """Show a stored run: its stages, its register, and what it cost."""
    # Reads the checkpoint and runs no stage, so the stand-in is used regardless
    # of what is configured. Inspecting a finished run should not need a key.
    graph, connection = compile_graph(FakePatternModel(), database)
    try:
        snapshot = graph.get_state(_config(run_id))
        if not snapshot.values:
            console.print(f"[yellow]No run with id {run_id}.[/yellow]")
            raise typer.Exit(code=1)

        if as_json:
            console.print_json(json.dumps(snapshot.values.get("register") or {}))
            return

        _show_stages(snapshot.values)
        _show_register(snapshot.values)
        _show_outcome(snapshot.values)
        console.print(f"[dim]next stage: {snapshot.next or '(finished)'}[/dim]")
    finally:
        connection.close()


# -- rendering --------------------------------------------------------------


def _show_stages(state: dict[str, Any]) -> None:
    entries = state.get("stage_log") or []
    if not entries:
        return

    table = Table(title="Stages", title_justify="left", box=None, pad_edge=False)
    table.add_column("stage")
    table.add_column("decision")
    table.add_column("detail", overflow="fold")

    notable = {"skip", "retry", "escalate", "discard", "conflict", "finding", "clean"}
    for entry in entries:
        decision = str(entry.get("decision", ""))
        style = "yellow" if decision in notable else ""
        source = entry.get("source")
        detail = str(entry.get("detail", ""))
        if source:
            detail = f"[{source}] {detail}"
        table.add_row(str(entry.get("stage", "")), Text(decision, style=style), detail)

    console.print(table)
    console.print()


def _show_register(state: dict[str, Any]) -> None:
    register = register_from(state.get("register") or {})
    if not register.obligations and not register.conflicts and not register.findings:
        return

    title = "Register as committed" if state.get("committed") else "Register as proposed"
    table = Table(title=title, title_justify="left", box=None)
    table.add_column("party")
    table.add_column("duty", overflow="fold")
    table.add_column("cited to")
    table.add_column("status")

    for row in register.obligations:
        span = row.support[0]
        table.add_row(
            row.party,
            row.duty,
            f"{span.source_id} · {span.locator}",
            "superseded" if row.superseded_by else "current",
        )

    console.print(table)
    console.print()

    _show_disagreements(register)


def _show_disagreements(register: Any) -> None:
    """The other two thirds of the register.

    Conflicts and findings are part of the deliverable, not commentary on it.
    Printing only the obligations leaves a reader counting them in the footer
    with no way to read what they say.
    """
    if register.conflicts:
        console.print("[bold]Conflicts[/bold]", style="yellow")
        console.print(
            "[dim]Reported, not resolved. Which side is right is a judgement about the "
            "business, so both are shown in full.[/dim]\n"
        )
        for conflict in register.conflicts:
            _show_conflict(conflict)

    if register.findings:
        table = Table(title="Findings", title_justify="left", box=None)
        table.add_column("rule")
        table.add_column("what failed", overflow="fold")
        table.add_column("cited to")

        for finding in register.findings:
            span = finding.support[0]
            table.add_row(
                finding.rule_id,
                finding.statement,
                f"{span.source_id} · {span.locator}",
            )
        console.print(table)
        console.print()


def _show_conflict(conflict: Any) -> None:
    """Both sides, each with its value, its place and its words.

    Enough to settle it away from the screen: which document, which clause,
    which line, and what it actually says.
    """
    table = Table(title=f"  {conflict.subject}", title_justify="left", box=None, padding=(0, 2))
    table.add_column("document")
    table.add_column("says")
    table.add_column("in its own words", overflow="fold")

    for fact in (conflict.left, conflict.right):
        span = fact.support[0]
        table.add_row(
            f"{span.source_id}\n[dim]{span.locator}[/dim]",
            f"[bold]{fact.value}[/bold]",
            f"[italic]{' '.join(span.quote.split())[:150]}[/italic]",
        )

    console.print(table)
    console.print()


def _review(proposals: list[Any], *, approve_all: bool) -> list[Any] | None:
    """Put every item in front of a person, one at a time."""
    import sys

    decided = []
    for position, proposal in enumerate(proposals, start=1):
        if approve_all:
            decided.append(
                proposal.with_decision(
                    Decision.APPROVE if proposal.changes_the_register else Decision.DEFER
                )
            )
            continue

        if not sys.stdin.isatty():
            console.print(
                "[red]The review gate needs an interactive terminal.[/red] "
                "Use --yes, or drive the run through the API."
            )
            return None

        _show_proposal(position, len(proposals), proposal)

        # Square brackets are Rich markup, so the key hints are escaped or they
        # are parsed as tags and stripped, leaving the prompt with no keys on it.
        choices = {"a": Decision.APPROVE, "r": Decision.REJECT, "d": Decision.DEFER}
        prompt = r"  \[a] approve   \[r] reject   \[d] defer   \[q] quit  > "

        while True:
            answer = console.input(prompt).strip().lower()
            if answer == "q":
                return None
            if answer in choices:
                decided.append(proposal.with_decision(choices[answer]))
                break
            # Asked again rather than defaulted. A stray keystroke that silently
            # deferred would look exactly like a decision that was made.
            console.print("[dim]  a, r, d or q[/dim]")

    return decided


def _show_proposal(position: int, total: int, proposal: Any) -> None:
    style = KIND_STYLES.get(proposal.kind, "white")

    body = Text()
    body.append(f"{proposal.summary}\n\n", style="bold")
    body.append(f"{proposal.reason}\n\n")
    body.append("Cited to\n", style="bold")
    for span in proposal.support[:3]:
        body.append(f"  {span.source_id} · {span.locator}\n", style="dim")
        body.append(f"  {span.quote.strip()[:160]}\n\n", style="italic")

    if not proposal.changes_the_register:
        body.append("Reported for a person. Approving this changes no row.\n", style="dim")

    console.print(
        Panel(
            body,
            title=f"[{style}]{str(proposal.kind).upper()}[/{style}]  item {position} of {total}",
            title_align="left",
            border_style=style,
        )
    )


def _show_delta(delta: RegisterDelta) -> None:
    """What the arrival moved, and what it demonstrably did not."""
    if delta.is_empty:
        console.print(f"[green]{delta.summary()}[/green]")
        return

    table = Table(title="What changed", title_justify="left", box=None)
    table.add_column("change")
    table.add_column("row", overflow="fold")
    table.add_column("because of")

    for row in delta.added:
        table.add_row(Text("added", style="green"), row.duty, row.support[0].source_id)
    for was, now in delta.changed:
        # What moved is often the status rather than the wording, so say which.
        if was.duty != now.duty:
            detail = f"{was.duty}\n  -> {now.duty}"
        elif was.superseded_by != now.superseded_by:
            detail = f"{now.duty}\n  -> now superseded by {now.superseded_by}"
        else:
            detail = now.duty
        table.add_row(Text("changed", style="yellow"), detail, now.support[0].source_id)
    for row in delta.removed:
        table.add_row(Text("removed", style="red"), row.duty, row.support[0].source_id)
    for conflict in delta.new_conflicts:
        table.add_row(Text("conflict", style="yellow"), conflict.explanation, "")
    for finding in delta.new_findings:
        table.add_row(Text("finding", style="magenta"), finding.statement[:80], finding.rule_id)

    console.print(table)
    console.print(
        f"\n[bold]{delta.untouched_count}[/bold] rows untouched, "
        f"verified by content digest  ·  because of: {', '.join(delta.because_of) or '—'}\n"
    )


def _show_outcome(state: dict[str, Any]) -> None:
    register = register_from(state.get("register") or {})
    cost = state.get("cost") or {}

    console.print(
        f"[bold]{len(register.current)}[/bold] current rows  ·  "
        f"[bold]{len(register.conflicts)}[/bold] conflicts  ·  "
        f"[bold]{len(register.findings)}[/bold] findings  ·  "
        f"digest [dim]{register.digest()}[/dim]"
    )
    if cost:
        console.print(
            f"[dim]model calls {cost.get('calls', 0)}  ·  "
            f"characters in {cost.get('input_chars', 0)}[/dim]"
        )


if __name__ == "__main__":
    app()
