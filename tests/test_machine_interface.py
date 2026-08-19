"""A program can drive the whole flow, and the gate is part of that flow.

Both surfaces are exercised: HTTP and MCP. What matters in each is the same —
approval is an explicit operation, decisions are per item, and there is no path
that reaches the register without them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
RULES = CORPUS / "rules" / "vendor-billing-checklist.yaml"
ARRIVAL = Path(__file__).resolve().parents[1] / "inbox" / "AMD-02-MSA-2026-014.md"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from analyst.api import app as module

    monkeypatch.setattr(module.settings, "checkpoint_path", tmp_path / "api.db")
    with TestClient(module.app) as test_client:
        yield test_client


def start(client: TestClient, run_id: str = "run-1") -> dict[str, Any]:
    response = client.post(
        "/runs",
        json={"corpus_dir": str(CORPUS), "rules_path": str(RULES), "run_id": run_id},
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


class TestHttpSurface:
    def test_health(self, client: TestClient) -> None:
        assert client.get("/health").json() == {"status": "ok"}

    def test_a_run_stops_at_the_gate(self, client: TestClient) -> None:
        summary = start(client)

        assert summary["stage"] == "commit"
        assert summary["awaiting_decision"] > 0
        assert summary["committed"] is False

    def test_the_stages_are_readable(self, client: TestClient) -> None:
        start(client)
        stages = client.get("/runs/run-1/stages").json()

        assert {e["stage"] for e in stages} >= {"ingest", "classify", "extract", "reconcile"}

    def test_every_item_carries_the_text_it_came_from(self, client: TestClient) -> None:
        start(client)
        proposals = client.get("/runs/run-1/proposals").json()

        assert proposals
        assert all(p["support"] for p in proposals)
        assert all(p["support"][0]["quote"].strip() for p in proposals)

    def test_committing_without_deciding_is_refused(self, client: TestClient) -> None:
        start(client)
        response = client.post("/runs/run-1/commit")

        assert response.status_code == 409
        assert response.json()["detail"]["undecided"]

    def test_a_program_can_drive_the_whole_flow(self, client: TestClient) -> None:
        """Start, read, decide item by item, commit. No human, no shortcut."""
        start(client)
        proposals = client.get("/runs/run-1/proposals").json()

        decisions = [{"proposal_id": p["proposal_id"], "decision": "approve"} for p in proposals]
        assert (
            client.post("/runs/run-1/decisions", json={"decisions": decisions}).status_code == 200
        )

        summary = client.post("/runs/run-1/commit").json()

        assert summary["committed"] is True
        assert summary["awaiting_decision"] == 0
        assert summary["obligations"] > 0

    def test_decisions_can_arrive_in_several_calls(self, client: TestClient) -> None:
        start(client)
        proposals = client.get("/runs/run-1/proposals").json()

        first = proposals[0]["proposal_id"]
        client.post(
            "/runs/run-1/decisions",
            json={"decisions": [{"proposal_id": first, "decision": "reject"}]},
        )
        summary = client.get("/runs/run-1").json()

        assert summary["awaiting_decision"] == len(proposals) - 1

    def test_rejecting_one_item_leaves_the_others(self, client: TestClient) -> None:
        start(client)
        proposals = client.get("/runs/run-1/proposals").json()
        rows = [p for p in proposals if p["kind"] in ("add_obligation", "supersede_obligation")]

        decisions = [
            {
                "proposal_id": p["proposal_id"],
                "decision": "reject" if p["proposal_id"] == rows[0]["proposal_id"] else "approve",
            }
            for p in proposals
        ]
        client.post("/runs/run-1/decisions", json={"decisions": decisions})
        summary = client.post("/runs/run-1/commit").json()

        assert summary["obligations"] == len(rows) - 1

    def test_the_same_corpus_is_not_read_twice(self, client: TestClient) -> None:
        first = client.post("/runs", json={"corpus_dir": str(CORPUS), "rules_path": str(RULES)})
        assert first.status_code == 201

        second = client.post("/runs", json={"corpus_dir": str(CORPUS), "rules_path": str(RULES)})

        assert second.status_code == 200
        assert second.json()["run_id"] == first.json()["run_id"]
        assert second.json()["cost"] == first.json()["cost"]

    def test_reading_it_again_is_available(self, client: TestClient) -> None:
        first = client.post("/runs", json={"corpus_dir": str(CORPUS), "rules_path": str(RULES)})
        again = client.post(
            "/runs", json={"corpus_dir": str(CORPUS), "rules_path": str(RULES), "fresh": True}
        )

        assert again.status_code == 201
        assert again.json()["run_id"] != first.json()["run_id"]

    def test_an_unknown_item_is_refused_by_name(self, client: TestClient) -> None:
        start(client)
        response = client.post(
            "/runs/run-1/decisions",
            json={"decisions": [{"proposal_id": "does-not-exist", "decision": "approve"}]},
        )

        assert response.status_code == 400
        assert "does-not-exist" in response.text

    def test_an_unknown_run_is_absent_rather_than_invented(self, client: TestClient) -> None:
        assert client.get("/runs/never-happened").status_code == 404

    def test_two_runs_stay_separate(self, client: TestClient) -> None:
        start(client, "run-a")
        start(client, "run-b")

        proposals = client.get("/runs/run-a/proposals").json()
        client.post(
            "/runs/run-a/decisions",
            json={
                "decisions": [
                    {"proposal_id": p["proposal_id"], "decision": "approve"} for p in proposals
                ]
            },
        )
        client.post("/runs/run-a/commit")

        assert client.get("/runs/run-a").json()["committed"] is True
        assert client.get("/runs/run-b").json()["committed"] is False

    def test_the_register_is_reported_with_its_citations(self, client: TestClient) -> None:
        start(client)
        register = client.get("/runs/run-1/register").json()

        assert register["obligations"]
        assert all(row["support"] for row in register["obligations"])


class TestAnArrivalIsAnUpdate:
    """A document arriving after the register exists costs what an arrival costs."""

    def commit_everything(self, client: TestClient) -> None:
        proposals = client.get("/runs/run-1/proposals").json()
        decisions = [{"proposal_id": p["proposal_id"], "decision": "approve"} for p in proposals]
        client.post("/runs/run-1/decisions", json={"decisions": decisions})
        client.post("/runs/run-1/commit")

    def test_a_program_can_fold_in_a_document_that_arrived(self, client: TestClient) -> None:
        start(client)
        self.commit_everything(client)
        spent = client.get("/runs/run-1").json()["cost"]["calls"]

        response = client.post("/runs/run-1/documents", json={"path": str(ARRIVAL)})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["delta"]["added"] or body["delta"]["changed"]
        assert body["delta"]["untouched"] > 0
        assert ARRIVAL.stem in body["delta"]["because_of"]

        after = client.get("/runs/run-1").json()["cost"]["calls"]
        assert after - spent < spent, "an arrival cost as much as reading the whole pile"

    def test_the_arrival_stops_at_the_gate(self, client: TestClient) -> None:
        start(client)
        self.commit_everything(client)
        client.post("/runs/run-1/documents", json={"path": str(ARRIVAL)})

        assert client.get("/runs/run-1").json()["awaiting_decision"] > 0
        assert client.post("/runs/run-1/commit").status_code == 409

    def test_the_same_document_is_not_read_twice(self, client: TestClient) -> None:
        start(client)
        self.commit_everything(client)
        client.post("/runs/run-1/documents", json={"path": str(ARRIVAL)})

        again = client.post("/runs/run-1/documents", json={"path": str(ARRIVAL)})

        assert again.status_code == 409

    def test_a_file_that_is_not_there_is_named_as_the_cause(self, client: TestClient) -> None:
        start(client)
        response = client.post("/runs/run-1/documents", json={"path": "nowhere/AMD-99.md"})

        assert response.status_code == 400
        assert "nowhere/AMD-99.md" in response.json()["detail"]


class TestMcpSurface:
    """The same operations as tools, so an agent can drive a run."""

    @pytest.fixture
    def tools(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        import analyst.mcp_server as module
        from analyst.adapters.fake_model import FakePatternModel
        from analyst.graph.build import compile_graph

        graph, connection = compile_graph(FakePatternModel(), tmp_path / "mcp.db")
        monkeypatch.setattr(module, "_graph", graph)
        try:
            yield module
        finally:
            connection.close()

    def test_it_exposes_the_whole_flow(self, tools: Any) -> None:
        import asyncio

        names = {t.name for t in asyncio.run(tools.mcp.list_tools())}

        assert names == {
            "start_run",
            "get_run",
            "get_stages",
            "list_proposals",
            "decide",
            "commit",
            "get_register",
            "ingest",
        }

    def test_the_same_corpus_is_not_read_twice(self, tools: Any) -> None:
        first = json.loads(tools.start_run(corpus_dir=str(CORPUS), rules_path=str(RULES)))
        second = json.loads(tools.start_run(corpus_dir=str(CORPUS), rules_path=str(RULES)))

        assert second["run_id"] == first["run_id"]
        assert second["cost"] == first["cost"]

        again = json.loads(
            tools.start_run(corpus_dir=str(CORPUS), rules_path=str(RULES), fresh=True)
        )
        assert again["run_id"] != first["run_id"]

    def test_an_agent_can_drive_a_run_to_completion(self, tools: Any) -> None:
        summary = json.loads(
            tools.start_run(corpus_dir=str(CORPUS), rules_path=str(RULES), run_id="run-m")
        )
        assert summary["stage"] == "commit"

        proposals = json.loads(tools.list_proposals("run-m"))
        assert proposals

        for item in proposals:
            tools.decide("run-m", item["proposal_id"], "approve")

        final = json.loads(tools.commit("run-m"))

        assert final["committed"] is True
        assert final["obligations"] > 0

    def test_an_agent_can_fold_in_a_document_that_arrived(self, tools: Any) -> None:
        tools.start_run(corpus_dir=str(CORPUS), rules_path=str(RULES), run_id="run-m")
        for item in json.loads(tools.list_proposals("run-m")):
            tools.decide("run-m", item["proposal_id"], "approve")
        tools.commit("run-m")

        answer = json.loads(tools.ingest("run-m", str(ARRIVAL)))

        assert answer["added"] or answer["changed"]
        assert answer["untouched"] > 0
        assert answer["run"]["awaiting_decision"] > 0, "an arrival must reach the gate too"

    def test_a_file_that_is_not_there_is_named_as_the_cause(self, tools: Any) -> None:
        tools.start_run(corpus_dir=str(CORPUS), rules_path=str(RULES), run_id="run-m")
        answer = json.loads(tools.ingest("run-m", "nowhere/AMD-99.md"))

        assert "nowhere/AMD-99.md" in answer["error"]

    def test_commit_is_refused_while_anything_is_undecided(self, tools: Any) -> None:
        tools.start_run(corpus_dir=str(CORPUS), rules_path=str(RULES), run_id="run-m")
        answer = json.loads(tools.commit("run-m"))

        assert "undecided" in answer

    def test_items_carry_their_citations(self, tools: Any) -> None:
        tools.start_run(corpus_dir=str(CORPUS), rules_path=str(RULES), run_id="run-m")
        proposals = json.loads(tools.list_proposals("run-m"))

        assert all(item["cited_to"] for item in proposals)
        assert all(item["cited_to"][0]["quote"].strip() for item in proposals)

    def test_a_bad_decision_is_refused_by_name(self, tools: Any) -> None:
        tools.start_run(corpus_dir=str(CORPUS), rules_path=str(RULES), run_id="run-m")
        answer = json.loads(tools.decide("run-m", "conflict:0", "maybe"))

        assert "error" in answer
