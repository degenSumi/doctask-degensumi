"""End-to-end behaviour of a run.

The claims this system makes about itself: that its stages can be watched, that
it survives being stopped, that nothing reaches the register without a decision,
that a program can drive all of it, and that it never asserts what it cannot
show in a document.

No key, no database, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from analyst.adapters.fake_model import FakePatternModel
from analyst.domain.models import Decision
from analyst.graph.build import compile_graph
from analyst.graph.serde import proposal_from, proposal_to, register_from

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
RULES = CORPUS / "rules" / "vendor-billing-checklist.yaml"


@pytest.fixture
def graph(tmp_path: Path) -> Any:
    compiled, connection = compile_graph(FakePatternModel(), tmp_path / "runs.db")
    yield compiled
    connection.close()


def config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def start(graph: Any, run_id: str = "run-1", corpus: Path = CORPUS) -> dict[str, Any]:
    return dict(
        graph.invoke(
            {"run_id": run_id, "corpus_dir": str(corpus), "rules_path": str(RULES)},
            config=config(run_id),
        )
    )


def decide_all(graph: Any, run_id: str, decision: Decision) -> None:
    values = graph.get_state(config(run_id)).values
    proposals = [proposal_from(p).with_decision(decision) for p in values["proposals"]]
    graph.update_state(config(run_id), {"proposals": [proposal_to(p) for p in proposals]})


class TestStagesCanBeWatched:
    def test_every_stage_reports_what_it_decided(self, graph: Any) -> None:
        state = start(graph)
        stages = {entry["stage"] for entry in state["stage_log"]}

        assert stages >= {"ingest", "classify", "extract", "reconcile", "examine", "propose"}

    def test_documents_are_classified_from_content_not_filename(self, graph: Any) -> None:
        state = start(graph)
        kinds = {sid: raw["kind"] for sid, raw in state["sources"].items()}

        assert kinds["MSA-2026-014"] == "contract"
        assert kinds["AMD-01-MSA-2026-014"] == "amendment"
        assert kinds["INV-1002"] == "invoice"

    def test_an_escalation_is_recorded_by_name(self, graph: Any) -> None:
        state = start(graph)
        escalations = [e for e in state["stage_log"] if e["decision"] == "escalate"]

        assert escalations
        assert escalations[0]["source"] == "NOTE-2026-05-14"

    def test_a_supersession_is_distinguished_from_a_conflict(self, graph: Any) -> None:
        """An amendment changing a term is the documents working as intended."""
        state = start(graph)
        decisions = [e["decision"] for e in state["stage_log"]]

        assert decisions.count("supersede") == 2
        assert decisions.count("conflict") == 2


class TestNothingIsAssertedWithoutASource:
    def test_every_register_row_is_cited(self, graph: Any) -> None:
        register = register_from(start(graph)["register"])

        assert register.obligations
        assert all(row.support for row in register.obligations)

    def test_every_citation_points_at_text_that_is_really_there(self, graph: Any) -> None:
        """The claim a span makes is checked against the document, not trusted."""
        from analyst.graph.serde import source_from

        state = start(graph)
        sources = {sid: source_from(raw) for sid, raw in state["sources"].items()}
        register = register_from(state["register"])

        for row in register.obligations:
            for span in row.support:
                assert span.verify(sources[span.source_id]), f"{row.obligation_id} cites stale text"

    def test_an_invoice_predating_an_amendment_is_not_reported_as_wrong(self, graph: Any) -> None:
        """INV-1001 is dated before the amendment took effect."""
        state = start(graph)
        conflicts = register_from(state["register"]).conflicts
        cited = {span.source_id for c in conflicts for span in c.support}

        assert "INV-1002" in cited
        assert "INV-1001" not in cited


class TestHostileDocuments:
    def test_a_document_giving_orders_is_quarantined(self, graph: Any) -> None:
        assert "NOTE-2026-05-14" in start(graph)["quarantined"]

    def test_it_still_reaches_a_person_as_a_finding(self, graph: Any) -> None:
        proposals = start(graph)["proposals"]
        kinds = [p["kind"] for p in proposals]

        assert "quarantine" in kinds

    def test_its_instructions_are_not_obeyed(self, graph: Any) -> None:
        """The note demands no discrepancy be reported on INV-1002."""
        register = register_from(start(graph)["register"])
        conflicts_on_1002 = [
            c for c in register.conflicts if any(s.source_id == "INV-1002" for s in c.support)
        ]

        assert conflicts_on_1002, "the hostile note suppressed a real conflict"


class TestTheGate:
    def test_a_run_stops_before_committing(self, graph: Any) -> None:
        start(graph)

        assert graph.get_state(config("run-1")).next == ("commit",)

    def test_nothing_is_committed_before_a_decision(self, graph: Any) -> None:
        state = start(graph)

        assert not state.get("committed")

    def test_committing_undecided_items_is_refused(self, graph: Any) -> None:
        start(graph)

        with pytest.raises(Exception, match="no decision"):
            graph.invoke(None, config=config("run-1"))

    def test_approving_writes_the_approved_rows(self, graph: Any) -> None:
        before = register_from(start(graph)["register"])
        decide_all(graph, "run-1", Decision.APPROVE)
        after = register_from(graph.invoke(None, config=config("run-1"))["register"])

        assert len(after.obligations) == len(before.obligations)

    def test_rejecting_everything_writes_no_rows(self, graph: Any) -> None:
        start(graph)
        decide_all(graph, "run-1", Decision.REJECT)
        after = register_from(graph.invoke(None, config=config("run-1"))["register"])

        assert after.obligations == ()

    def test_rejecting_one_leaves_the_rest(self, graph: Any) -> None:
        state = start(graph)
        proposals = [proposal_from(p) for p in state["proposals"]]
        rows = [p for p in proposals if p.changes_the_register]
        assert len(rows) > 1

        rejected = rows[0].proposal_id
        updated = [
            p.with_decision(Decision.REJECT if p.proposal_id == rejected else Decision.APPROVE)
            for p in proposals
        ]
        graph.update_state(config("run-1"), {"proposals": [proposal_to(p) for p in updated]})
        after = register_from(graph.invoke(None, config=config("run-1"))["register"])

        assert len(after.obligations) == len(rows) - 1

    def test_findings_and_conflicts_survive_the_commit(self, graph: Any) -> None:
        """They are reported, not applied, so a decision on them changes no row."""
        start(graph)
        decide_all(graph, "run-1", Decision.REJECT)
        after = register_from(graph.invoke(None, config=config("run-1"))["register"])

        assert after.conflicts
        assert after.findings


class TestSurvivingBeingStopped:
    def test_state_is_on_disk_at_the_gate(self, graph: Any) -> None:
        start(graph)
        stored = graph.get_state(config("run-1")).values

        assert stored["sources"]
        assert stored["proposals"]

    def test_resuming_redoes_no_model_work(self, tmp_path: Path) -> None:
        """The claim that matters: a stopped run continues rather than restarts."""
        first_model = FakePatternModel()
        first, connection = compile_graph(first_model, tmp_path / "runs.db")
        first.invoke(
            {"run_id": "run-1", "corpus_dir": str(CORPUS), "rules_path": str(RULES)},
            config=config("run-1"),
        )
        spent = first_model.cost.calls
        connection.close()

        # A new process, a new model, the same checkpoint.
        second_model = FakePatternModel()
        second, connection = compile_graph(second_model, tmp_path / "runs.db")
        try:
            decide_all(second, "run-1", Decision.APPROVE)
            second.invoke(None, config=config("run-1"))

            assert spent > 0
            assert second_model.cost.calls == 0
        finally:
            connection.close()

    def test_the_run_reports_what_the_first_attempt_spent(self, tmp_path: Path) -> None:
        first, connection = compile_graph(FakePatternModel(), tmp_path / "runs.db")
        first.invoke(
            {"run_id": "run-1", "corpus_dir": str(CORPUS), "rules_path": str(RULES)},
            config=config("run-1"),
        )
        connection.close()

        second, connection = compile_graph(FakePatternModel(), tmp_path / "runs.db")
        try:
            decide_all(second, "run-1", Decision.APPROVE)
            final = second.invoke(None, config=config("run-1"))

            assert final["cost"]["calls"] > 0
        finally:
            connection.close()


class TestConcurrentRuns:
    def test_two_runs_stay_separate(self, graph: Any) -> None:
        start(graph, "run-a")
        start(graph, "run-b")

        decide_all(graph, "run-a", Decision.APPROVE)
        decide_all(graph, "run-b", Decision.REJECT)
        graph.invoke(None, config=config("run-a"))
        graph.invoke(None, config=config("run-b"))

        a = register_from(graph.get_state(config("run-a")).values["register"])
        b = register_from(graph.get_state(config("run-b")).values["register"])

        assert a.obligations
        assert b.obligations == ()

    def test_a_run_is_reproducible(self, graph: Any) -> None:
        """Two runs over the same corpus produce the same register."""
        first = register_from(start(graph, "run-a")["register"])
        second = register_from(start(graph, "run-b")["register"])

        assert first.digest() == second.digest()


class TestEmptyAndDegradedCorpora:
    def test_an_empty_folder_produces_nothing_rather_than_failing(
        self, graph: Any, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        state = start(graph, "run-empty", empty)

        assert state["sources"] == {}
        assert state["proposals"] == []

    def test_an_unreadable_format_is_skipped_by_name(self, graph: Any, tmp_path: Path) -> None:
        folder = tmp_path / "mixed"
        folder.mkdir()
        (folder / "notes.md").write_text("# FILE NOTE\n\nNothing of interest here.\n")
        (folder / "sheet.xlsx").write_bytes(b"not really a spreadsheet")

        state = start(graph, "run-mixed", folder)

        assert "notes" in state["sources"]
        assert "sheet" not in state["sources"]

    def test_a_clean_corpus_reports_no_findings(self, graph: Any, tmp_path: Path) -> None:
        """The honest empty result, rather than a finding invented to fill it."""
        folder = tmp_path / "clean"
        folder.mkdir()
        (folder / "MSA.md").write_text(
            (CORPUS / "MSA-2026-014.md").read_text(encoding="utf-8"), encoding="utf-8"
        )

        register = register_from(start(graph, "run-clean", folder)["register"])

        assert register.findings == ()
        assert register.conflicts == ()
        assert register.obligations
