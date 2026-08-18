"""A document arriving after a register exists.

The claim: an arrival produces a focused update, not a rewrite. What it did not
affect stays exactly as it was, and that is checked rather than asserted. What it
did affect is attributable to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from analyst.adapters.fake_model import FakePatternModel
from analyst.app.update import NotUpdatable, ingest
from analyst.domain.models import Decision
from analyst.graph.build import compile_graph
from analyst.graph.serde import proposal_from, proposal_to, register_from

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
RULES = CORPUS / "rules" / "vendor-billing-checklist.yaml"
ARRIVAL = ROOT / "inbox" / "AMD-02-MSA-2026-014.md"


def config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


@pytest.fixture
def finished(tmp_path: Path) -> Any:
    """A run that has read the corpus, been approved, and committed."""
    model = FakePatternModel()
    graph, connection = compile_graph(model, tmp_path / "runs.db")

    graph.invoke(
        {"run_id": "run-1", "corpus_dir": str(CORPUS), "rules_path": str(RULES)},
        config=config("run-1"),
    )
    values = graph.get_state(config("run-1")).values
    decided = [proposal_from(p).with_decision(Decision.APPROVE) for p in values["proposals"]]
    graph.update_state(config("run-1"), {"proposals": [proposal_to(p) for p in decided]})
    graph.invoke(None, config=config("run-1"))

    yield graph, model
    connection.close()


class TestAnArrival:
    def test_it_changes_only_what_it_touches(self, finished: Any) -> None:
        graph, _ = finished
        _, delta = ingest(graph, "run-1", ARRIVAL)

        assert delta.added
        assert delta.untouched_count > 0
        assert not delta.removed

    def test_untouched_rows_are_verified_not_assumed(self, finished: Any) -> None:
        """Every unchanged row has the same content digest it had before."""
        graph, _ = finished
        before = register_from(graph.get_state(config("run-1")).values["register"])
        before_digests = {row.obligation_id: row.digest() for row in before.obligations}

        _, delta = ingest(graph, "run-1", ARRIVAL)

        for row in delta.unchanged:
            assert row.digest() == before_digests[row.obligation_id]

    def test_what_changed_is_attributable_to_the_arrival(self, finished: Any) -> None:
        graph, _ = finished
        _, delta = ingest(graph, "run-1", ARRIVAL)

        assert "AMD-02-MSA-2026-014" in delta.because_of

    def test_the_new_term_supersedes_the_old_one(self, finished: Any) -> None:
        graph, _ = finished
        state, _ = ingest(graph, "run-1", ARRIVAL)
        register = register_from(state["register"])

        notice = [r for r in register.obligations if "notice to terminate" in r.duty]
        current = [r for r in notice if r.is_current]
        superseded = [r for r in notice if not r.is_current]

        assert len(current) == 1
        assert "thirty" in current[0].duty
        assert superseded and superseded[0].superseded_by == "AMD-02-MSA-2026-014"

    def test_an_update_costs_like_an_update(self, finished: Any) -> None:
        """One document arriving must not cost a pass over the whole pile."""
        graph, model = finished
        full_run = model.cost.calls

        before = model.cost.calls
        ingest(graph, "run-1", ARRIVAL)
        update_cost = model.cost.calls - before

        assert update_cost > 0, "the arrival was not read at all"
        assert update_cost * 3 < full_run, (
            f"the update cost {update_cost} calls against {full_run} for the full run"
        )

    def test_it_asks_only_about_what_moved(self, finished: Any) -> None:
        """A person's attention costs what the arrival costs, like the model's.

        Putting every row back in front of someone is where a tired reviewer
        waves through the one item that actually changed.
        """
        graph, _ = finished
        state, delta = ingest(graph, "run-1", ARRIVAL)

        proposals = [proposal_from(p) for p in state["proposals"]]
        outstanding = [p for p in proposals if not p.is_decided]

        assert outstanding, "the arrival produced nothing to decide"
        assert len(outstanding) < len(proposals) / 2, (
            f"{len(outstanding)} of {len(proposals)} items were put up again, "
            f"though only {len(delta.added) + len(delta.changed)} rows moved"
        )

    def test_a_row_that_changed_is_asked_about_again(self, finished: Any) -> None:
        """Carrying a decision forward is matched on content, not on identifier."""
        graph, _ = finished
        state, delta = ingest(graph, "run-1", ARRIVAL)

        proposals = [proposal_from(p) for p in state["proposals"]]
        undecided_ids = {p.proposal_id for p in proposals if not p.is_decided}
        moved = {
            f"row:{row.obligation_id}" for row in (*delta.added, *(n for _, n in delta.changed))
        }

        assert moved <= undecided_ids, (
            f"a row that moved was carried forward: {moved - undecided_ids}"
        )

    def test_findings_about_other_documents_survive(self, finished: Any) -> None:
        """Nobody re-examined INV-1002, so its finding must not disappear."""
        graph, _ = finished
        before = register_from(graph.get_state(config("run-1")).values["register"])

        state, _ = ingest(graph, "run-1", ARRIVAL)
        after = register_from(state["register"])

        assert {f.rule_id for f in before.findings} <= {f.rule_id for f in after.findings}

    def test_conflicts_found_earlier_are_still_reported(self, finished: Any) -> None:
        graph, _ = finished
        state, _ = ingest(graph, "run-1", ARRIVAL)

        assert register_from(state["register"]).conflicts

    def test_the_run_stops_at_the_gate_again(self, finished: Any) -> None:
        """An arrival implies changes, and those need deciding like any others."""
        graph, _ = finished
        ingest(graph, "run-1", ARRIVAL)

        assert graph.get_state(config("run-1")).next == ("commit",)

    def test_the_same_document_twice_is_refused(self, finished: Any) -> None:
        graph, _ = finished
        ingest(graph, "run-1", ARRIVAL)

        with pytest.raises(NotUpdatable, match="already part of run"):
            ingest(graph, "run-1", ARRIVAL)

    def test_an_unknown_run_is_refused(self, finished: Any) -> None:
        graph, _ = finished

        with pytest.raises(NotUpdatable, match="no run with id"):
            ingest(graph, "never-happened", ARRIVAL)


class TestAnArrivalThatChangesNothing:
    def test_a_document_with_no_terms_moves_nothing(self, finished: Any, tmp_path: Path) -> None:
        graph, _ = finished
        empty = tmp_path / "COVER-NOTE.md"
        empty.write_text(
            "# FILE NOTE\n\nPlease find the signed copies attached for your records.\n",
            encoding="utf-8",
        )

        _, delta = ingest(graph, "run-1", empty)

        assert delta.is_empty
        assert "nothing changed" in delta.summary()
        assert delta.untouched_count > 0
