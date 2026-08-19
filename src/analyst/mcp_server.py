"""MCP interface.

The same operations as the HTTP surface, exposed as tools so an agent can drive
a whole run: read the pile, look at what was proposed, decide item by item,
commit, and fold in a document that arrived afterwards.

The gate is part of that flow rather than an exception to it. `commit` refuses
while anything is undecided, so an agent driving this has to make the call
explicitly, item by item, exactly as a person at a terminal would. There is no
tool that commits without decisions and no argument that skips the gate.

Run:
    uv run python -m analyst.mcp_server
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from analyst.adapters.loader import UnsupportedFormat
from analyst.app import update
from analyst.app.runs import run_id_for
from analyst.domain.models import Decision
from analyst.graph.build import compile_graph
from analyst.graph.serde import proposal_from, proposal_to
from analyst.settings import Settings

settings = Settings()
mcp = MCPServer(
    "analyst",
    instructions=(
        "Reads a pile of related documents and keeps one cited register current. "
        "Start a run, look at what it proposed, decide on each item, then commit. "
        "Nothing reaches the register until every item has been decided."
    ),
)

_graph, _connection = compile_graph(settings.build_model(), settings.checkpoint_path)


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def _state(run_id: str) -> Any:
    snapshot = _graph.get_state(_config(run_id))
    if not snapshot.values:
        raise ValueError(f"no run with id {run_id}")
    return snapshot


def _summary(run_id: str, snapshot: Any) -> dict[str, Any]:
    values = snapshot.values
    register = values.get("register") or {}
    proposals = values.get("proposals") or []
    return {
        "run_id": run_id,
        "stage": snapshot.next[0] if snapshot.next else "finished",
        "awaiting_decision": sum(1 for p in proposals if not p.get("decision")),
        "obligations": len(register.get("obligations") or []),
        "conflicts": len(register.get("conflicts") or []),
        "findings": len(register.get("findings") or []),
        "committed": bool(values.get("committed")),
        "cost": values.get("cost") or {},
    }


@mcp.tool()
def start_run(
    corpus_dir: str = "corpus",
    rules_path: str = "corpus/rules/vendor-billing-checklist.yaml",
    run_id: str | None = None,
    fresh: bool = False,
) -> str:
    """Read a folder of documents and stop at the human gate.

    Returns a summary. Nothing has been written to the register: every item is
    waiting for a decision.

    A run is named after what it reads, so calling this twice on the same corpus
    returns the run that already read it rather than reading it again. Pass
    `fresh` to read it again as a separate run.
    """
    folder = Path(corpus_dir)
    if not folder.exists():
        return json.dumps({"error": f"no such folder: {corpus_dir}"})

    identifier = run_id or run_id_for(folder, Path(rules_path), fresh=fresh)

    # Re-entering a run that has already read this corpus would start at the
    # first stage and pay for the same documents a second time, and would
    # discard whatever decisions had been recorded against them.
    existing = _graph.get_state(_config(identifier))
    if existing.values:
        return json.dumps(_summary(identifier, existing), indent=2)

    _graph.invoke(
        {"run_id": identifier, "corpus_dir": str(folder), "rules_path": rules_path},
        config=_config(identifier),
    )
    return json.dumps(_summary(identifier, _graph.get_state(_config(identifier))), indent=2)


@mcp.tool()
def get_run(run_id: str) -> str:
    """Where a run has got to, and what it has produced so far."""
    return json.dumps(_summary(run_id, _state(run_id)), indent=2)


@mcp.tool()
def get_stages(run_id: str) -> str:
    """What the run decided at each stage, in order.

    Includes the decisions that changed its path: documents skipped, documents
    read a second time, and documents quarantined for addressing the system.
    """
    return json.dumps(_state(run_id).values.get("stage_log") or [], indent=2)


@mcp.tool()
def list_proposals(run_id: str) -> str:
    """Every item awaiting a decision.

    Each carries the exact source text it came from, so a decision can be made
    on evidence rather than on a summary.
    """
    proposals = _state(run_id).values.get("proposals") or []
    return json.dumps(
        [
            {
                "proposal_id": p["proposal_id"],
                "kind": p["kind"],
                "summary": p["summary"],
                "reason": p["reason"],
                "decided_by": p["decided_by"],
                "decision": p.get("decision"),
                "cited_to": [
                    {
                        "source": s["source_id"],
                        "locator": s["locator"],
                        "quote": s["quote"][:300],
                    }
                    for s in p["support"][:3]
                ],
            }
            for p in proposals
        ],
        indent=2,
    )


@mcp.tool()
def decide(run_id: str, proposal_id: str, decision: str) -> str:
    """Record a decision on one item: approve, reject or defer.

    Items are decided individually. Rejecting one leaves every other item as it
    was.
    """
    try:
        chosen = Decision(decision)
    except ValueError:
        return json.dumps({"error": f"decision must be approve, reject or defer, not {decision!r}"})

    snapshot = _state(run_id)
    proposals = [proposal_from(p) for p in snapshot.values.get("proposals") or []]
    if proposal_id not in {p.proposal_id for p in proposals}:
        return json.dumps({"error": f"no item with id {proposal_id}"})

    updated = [p.with_decision(chosen) if p.proposal_id == proposal_id else p for p in proposals]
    _graph.update_state(_config(run_id), {"proposals": [proposal_to(p) for p in updated]})

    return json.dumps(_summary(run_id, _graph.get_state(_config(run_id))), indent=2)


@mcp.tool()
def commit(run_id: str) -> str:
    """Apply the decisions and finish the run.

    Refused while any item is undecided, and the reply names the ones still
    outstanding. This is the gate: it is not possible to reach the register
    without having decided on everything.
    """
    snapshot = _state(run_id)
    proposals = snapshot.values.get("proposals") or []
    undecided = [p["proposal_id"] for p in proposals if not p.get("decision")]
    if undecided:
        return json.dumps(
            {
                "error": "every item must be decided before the run can commit",
                "undecided": undecided,
            },
            indent=2,
        )
    if not snapshot.next:
        return json.dumps({"error": "this run has already finished"})

    _graph.invoke(None, config=_config(run_id))
    return json.dumps(_summary(run_id, _graph.get_state(_config(run_id))), indent=2)


@mcp.tool()
def ingest(run_id: str, path: str) -> str:
    """Add a document that arrived to a run that has already produced a register.

    Not a re-run: sources already read are not read again and rules already
    checked are not checked again, so an arrival costs what an arrival costs.
    The reply names what moved and counts what did not, compared by content
    digest rather than asserted. The run is left at the gate, so call
    `list_proposals` and `decide` next.
    """
    document = Path(path)
    if not document.exists():
        return json.dumps({"error": f"no such file: {path}"})

    try:
        _, delta = update.ingest(_graph, run_id, document)
    except (update.NotUpdatable, UnsupportedFormat) as error:
        return json.dumps({"error": str(error)})

    return json.dumps(
        {
            "summary": delta.summary(),
            "added": [o.duty for o in delta.added],
            "changed": [after.duty for _, after in delta.changed],
            "removed": [o.duty for o in delta.removed],
            "untouched": delta.untouched_count,
            "new_conflicts": len(delta.new_conflicts),
            "new_findings": len(delta.new_findings),
            "because_of": list(delta.because_of),
            "run": _summary(run_id, _graph.get_state(_config(run_id))),
        },
        indent=2,
    )


@mcp.tool()
def get_register(run_id: str) -> str:
    """The deliverable: obligations, conflicts and findings, each cited."""
    register = _state(run_id).values.get("register") or {}
    return json.dumps(
        {
            "obligations": [
                {
                    "party": o["party"],
                    "duty": o["duty"],
                    "status": "superseded" if o["superseded_by"] else "current",
                    "cited_to": [f"{s['source_id']} · {s['locator']}" for s in o["support"][:2]],
                }
                for o in register.get("obligations") or []
            ],
            "conflicts": [
                {"subject": c["subject"], "explanation": c["explanation"]}
                for c in register.get("conflicts") or []
            ],
            "findings": [
                {"rule_id": f["rule_id"], "severity": f["severity"], "statement": f["statement"]}
                for f in register.get("findings") or []
            ],
        },
        indent=2,
    )


def main() -> None:
    try:
        mcp.run()
    finally:
        _connection.close()


if __name__ == "__main__":
    main()
