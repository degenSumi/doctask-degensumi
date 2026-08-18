"""Adding a document to a run that has already produced a register.

An arrival is not a re-run. The document is added to the run's own state and the
graph is re-entered, where each stage skips what it has already done: sources
already read are not read again, facts already grounded are not extracted again,
and rules already checked against a source are not checked again. What the
arrival costs is what the arrival costs.

The register before the update is kept, so the change can be shown row by row
rather than described. Rows the new document did not affect are compared by
content digest and reported as untouched — a claim that is checked, not made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from analyst.adapters.loader import read_text
from analyst.domain.delta import RegisterDelta, compare
from analyst.domain.models import Source
from analyst.graph.serde import register_from, source_to


class NotUpdatable(Exception):
    """Raised when a run cannot take an arrival in its current state."""


def config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def ingest(graph: Any, run_id: str, path: Path) -> tuple[dict[str, Any], RegisterDelta]:
    """Add one document to an existing run and report what moved.

    Returns the new state and the difference. The run is left at the gate again:
    an arrival produces items to decide on, exactly as a first run does, because
    nothing a new document implies should reach the register unreviewed.
    """
    snapshot = graph.get_state(config(run_id))
    if not snapshot.values:
        raise NotUpdatable(f"no run with id {run_id}")

    values = snapshot.values
    source_id = path.stem

    if source_id in (values.get("sources") or {}):
        raise NotUpdatable(
            f"{path.name} is already part of run {run_id}. "
            "Rename the file or start a new run to read it again."
        )

    text = read_text(path)
    before = register_from(values.get("register") or {})

    sources = dict(values.get("sources") or {})
    sources[source_id] = source_to(Source(source_id=source_id, filename=path.name, text=text))

    # Re-enter at classification. Ingest is skipped because the document is
    # already in state, and every stage after it filters on what it has done.
    graph.update_state(
        config(run_id),
        {
            "sources": sources,
            "previous_register": values.get("register") or {},
            "proposals": [],
        },
        as_node="ingest",
    )

    after_values = dict(graph.invoke(None, config=config(run_id)))
    after = register_from(after_values.get("register") or {})

    return after_values, compare(before, after)
