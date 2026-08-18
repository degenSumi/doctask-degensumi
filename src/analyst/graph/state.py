"""The state carried between stages.

Plain JSON-compatible data, because it is written to a checkpointer after every
stage and read back on resume. Domain objects are rebuilt inside a node, used,
and converted back before the node returns.

`stage_log` is the record of what happened and why. It exists so a run can be
watched rather than inferred: every stage appends what it decided, and the
decisions that change the path say so by name.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def append(existing: list[Any] | None, incoming: list[Any] | None) -> list[Any]:
    """Reducer that accumulates rather than replaces.

    Applied to the stage log and the skip list so a resumed run keeps the record
    of what happened before it stopped.
    """
    return [*(existing or []), *(incoming or [])]


def merge(existing: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    return {**(existing or {}), **(incoming or {})}


def add_up(existing: dict[str, int] | None, incoming: dict[str, int] | None) -> dict[str, int]:
    """Reducer that sums counters across stages.

    Applied to cost so the figure is the run's total, not the last stage's.
    A resumed run keeps what the first attempt spent, which is the honest
    number: that work was paid for whether or not the process survived.
    """
    total = dict(existing or {})
    for key, value in (incoming or {}).items():
        total[key] = total.get(key, 0) + value
    return total


class RunState(TypedDict, total=False):
    run_id: str
    corpus_dir: str
    rules_path: str

    sources: dict[str, Any]
    """source_id -> serialised Source."""

    facts: list[dict[str, Any]]
    quarantined: list[str]
    conflicts: list[dict[str, Any]]
    supersessions: list[dict[str, Any]]
    register: dict[str, Any]
    proposals: list[dict[str, Any]]

    stage_log: Annotated[list[dict[str, Any]], append]
    skipped: Annotated[list[dict[str, Any]], append]
    retries: Annotated[dict[str, Any], merge]

    extracted: list[str]
    """Sources already read for facts. Skipped on a later pass."""

    findings: list[dict[str, Any]]
    """Single-document findings, kept across passes.

    They live here as well as in the register because the register is rebuilt
    from facts on every reconcile, and a finding about a document nobody
    re-examined would otherwise disappear.
    """

    examined: list[str]
    """Sources already checked against the rules.

    An update re-checks only what arrived. Without this a new document would
    cost a full pass over every rule and every source, which is what "an update
    should cost like an update" rules out.
    """

    previous_register: dict[str, Any]
    """The register as it stood before this update, kept so the change can be
    shown rather than described."""

    cost: Annotated[dict[str, int], add_up]
    committed: bool


def log_entry(stage: str, decision: str, detail: str = "", **extra: Any) -> dict[str, Any]:
    """One line of the visible record of a run."""
    return {"stage": stage, "decision": decision, "detail": detail, **extra}
