"""HTTP interface.

Everything a person can do at a terminal, another program can do here, and that
includes the gate. Approval is its own call: a run pauses before committing and
stays paused until decisions arrive, whoever sends them. There is no path that
commits by default and no flag that skips the gate.

Decisions are per item. A caller that decides on some items and not others gets
an error naming the ones it missed, rather than a partial commit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from analyst.adapters.loader import UnsupportedFormat
from analyst.app import update
from analyst.app.runs import run_id_for
from analyst.domain.delta import RegisterDelta
from analyst.domain.models import Decision
from analyst.graph.build import compile_graph
from analyst.graph.serde import proposal_from, proposal_to
from analyst.settings import Settings

settings = Settings()
_graph: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _graph
    graph, connection = compile_graph(settings.build_model(), settings.checkpoint_path)
    _graph = graph
    try:
        yield
    finally:
        connection.close()


app = FastAPI(
    title="analyst",
    summary="Reads a pile of related documents and keeps one cited register current.",
    lifespan=lifespan,
)


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def _require(run_id: str) -> Any:
    snapshot = _graph.get_state(_config(run_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"no run with id {run_id}")
    return snapshot


# -- request and response shapes -------------------------------------------


class StartRun(BaseModel):
    corpus_dir: str = Field(default="corpus", description="Folder of documents to read.")
    rules_path: str = Field(
        default="corpus/rules/vendor-billing-checklist.yaml",
        description="Rules file to check the corpus against.",
    )
    run_id: str | None = Field(default=None, description="Supply to name the run yourself.")
    fresh: bool = Field(default=False, description="Read the corpus again as a separate run.")


class Arrival(BaseModel):
    path: str = Field(description="The document that arrived, readable by this process.")


class DeltaOut(BaseModel):
    summary: str
    added: list[str]
    changed: list[str]
    removed: list[str]
    untouched: int = Field(description="Rows compared by content digest and found identical.")
    new_conflicts: int
    new_findings: int
    because_of: list[str]


class Ingested(BaseModel):
    delta: DeltaOut
    run: RunSummary


class DecisionIn(BaseModel):
    proposal_id: str
    decision: Literal["approve", "reject", "defer"]


class Decide(BaseModel):
    decisions: list[DecisionIn]


class RunSummary(BaseModel):
    run_id: str
    stage: str
    awaiting_decision: int
    obligations: int
    conflicts: int
    findings: int
    committed: bool
    cost: dict[str, int]


def _delta_out(delta: RegisterDelta) -> DeltaOut:
    return DeltaOut(
        summary=delta.summary(),
        added=[o.duty for o in delta.added],
        changed=[after.duty for _, after in delta.changed],
        removed=[o.duty for o in delta.removed],
        untouched=delta.untouched_count,
        new_conflicts=len(delta.new_conflicts),
        new_findings=len(delta.new_findings),
        because_of=list(delta.because_of),
    )


def _summary(run_id: str, snapshot: Any) -> RunSummary:
    values = snapshot.values
    register = values.get("register") or {}
    proposals = values.get("proposals") or []
    return RunSummary(
        run_id=run_id,
        stage=snapshot.next[0] if snapshot.next else "finished",
        awaiting_decision=sum(1 for p in proposals if not p.get("decision")),
        obligations=len(register.get("obligations") or []),
        conflicts=len(register.get("conflicts") or []),
        findings=len(register.get("findings") or []),
        committed=bool(values.get("committed")),
        cost=values.get("cost") or {},
    )


# -- endpoints --------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=RunSummary, status_code=201)
def start_run(body: StartRun, response: Response) -> RunSummary:
    """Read the corpus and stop at the gate.

    Returns once every item is prepared and awaiting a decision. Nothing has
    been written to the register at this point.

    A run is named after what it reads, so posting the same corpus twice returns
    the run that already read it, with 200 rather than 201. Pass `fresh` to read
    it again as a separate run.
    """
    corpus = Path(body.corpus_dir)
    if not corpus.exists():
        raise HTTPException(status_code=400, detail=f"no such folder: {corpus}")

    run_id = body.run_id or run_id_for(corpus, Path(body.rules_path), fresh=body.fresh)

    # Re-entering a run that has already read this corpus would start at the
    # first stage and pay for the same documents a second time, and would
    # discard whatever decisions had been recorded against them.
    existing = _graph.get_state(_config(run_id))
    if existing.values:
        response.status_code = 200
        return _summary(run_id, existing)

    _graph.invoke(
        {"run_id": run_id, "corpus_dir": str(corpus), "rules_path": body.rules_path},
        config=_config(run_id),
    )
    return _summary(run_id, _graph.get_state(_config(run_id)))


@app.get("/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: str) -> RunSummary:
    return _summary(run_id, _require(run_id))


@app.get("/runs/{run_id}/stages")
def get_stages(run_id: str) -> list[dict[str, Any]]:
    """What the run decided at each stage, in order."""
    return list(_require(run_id).values.get("stage_log") or [])


@app.get("/runs/{run_id}/proposals")
def get_proposals(run_id: str) -> list[dict[str, Any]]:
    """Every item awaiting a decision, each carrying the text it came from."""
    return list(_require(run_id).values.get("proposals") or [])


@app.get("/runs/{run_id}/register")
def get_register(run_id: str) -> dict[str, Any]:
    """The deliverable as it currently stands."""
    return dict(_require(run_id).values.get("register") or {})


@app.post("/runs/{run_id}/documents", response_model=Ingested)
def ingest(run_id: str, body: Arrival) -> Ingested:
    """Add one document that arrived to a run that has already produced a register.

    Not a re-run. Sources already read are not read again and rules already
    checked are not checked again, so an arrival costs what an arrival costs.
    The reply names what moved and counts what did not, compared by content
    digest rather than asserted.

    The run is left at the gate, exactly as a first run is: nothing a new
    document implies reaches the register without a decision.
    """
    path = Path(body.path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"no such file: {path}")

    try:
        _, delta = update.ingest(_graph, run_id, path)
    except update.NotUpdatable as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except UnsupportedFormat as error:
        raise HTTPException(status_code=415, detail=str(error)) from error

    return Ingested(
        delta=_delta_out(delta),
        run=_summary(run_id, _graph.get_state(_config(run_id))),
    )


@app.post("/runs/{run_id}/decisions", response_model=RunSummary)
def decide(run_id: str, body: Decide) -> RunSummary:
    """Record decisions on individual items.

    Items not named keep whatever decision they already had, so a caller may
    decide in several calls. Nothing is committed by this endpoint.
    """
    snapshot = _require(run_id)
    proposals = [proposal_from(p) for p in snapshot.values.get("proposals") or []]
    known = {p.proposal_id for p in proposals}

    unknown = [d.proposal_id for d in body.decisions if d.proposal_id not in known]
    if unknown:
        raise HTTPException(status_code=400, detail=f"no such item(s): {unknown}")

    chosen = {d.proposal_id: Decision(d.decision) for d in body.decisions}
    updated = [
        p.with_decision(chosen[p.proposal_id]) if p.proposal_id in chosen else p for p in proposals
    ]

    _graph.update_state(_config(run_id), {"proposals": [proposal_to(p) for p in updated]})
    return _summary(run_id, _graph.get_state(_config(run_id)))


@app.post("/runs/{run_id}/commit", response_model=RunSummary)
def commit(run_id: str) -> RunSummary:
    """Apply the decisions and finish the run.

    Refused while any item is undecided, because a commit that silently skipped
    them would put rows in the register nobody looked at.
    """
    snapshot = _require(run_id)
    proposals = snapshot.values.get("proposals") or []
    undecided = [p["proposal_id"] for p in proposals if not p.get("decision")]
    if undecided:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "every item must be decided before the run can commit",
                "undecided": undecided,
            },
        )

    if not snapshot.next:
        raise HTTPException(status_code=409, detail="this run has already finished")

    _graph.invoke(None, config=_config(run_id))
    return _summary(run_id, _graph.get_state(_config(run_id)))
