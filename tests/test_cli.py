"""Which run an invocation lands on.

A run is identified by what it reads, so pointing at the same corpus twice
continues the work rather than paying for it again. Starting over is available,
but it is asked for.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from analyst.app.runs import run_id_for
from analyst.cli import app

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
RULES = CORPUS / "rules" / "vendor-billing-checklist.yaml"

runner = CliRunner()


def invoke(corpus: Path, database: Path, *extra: str) -> str:
    result = runner.invoke(
        app,
        ["run", str(corpus), "--rules", str(RULES), "--database", str(database), *extra],
    )
    assert result.exit_code == 0, result.output
    return result.output


class TestTheRunIsIdentifiedByWhatItReads:
    def test_the_same_corpus_and_rules_name_the_same_run(self) -> None:
        assert run_id_for(CORPUS, RULES, fresh=False) == run_id_for(CORPUS, RULES, fresh=False)

    def test_the_path_is_the_folder_not_the_spelling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(CORPUS.parent)
        assert run_id_for(Path("corpus"), RULES, fresh=False) == run_id_for(
            CORPUS, RULES, fresh=False
        )

    def test_different_rules_are_a_different_question(self, tmp_path: Path) -> None:
        other = tmp_path / "other-rules.yaml"
        other.write_text("rules: []\n")

        assert run_id_for(CORPUS, RULES, fresh=False) != run_id_for(CORPUS, other, fresh=False)

    def test_starting_over_is_a_separate_run(self) -> None:
        assert run_id_for(CORPUS, RULES, fresh=True) != run_id_for(CORPUS, RULES, fresh=False)


class TestASecondInvocation:
    def test_a_finished_run_is_reported_not_repeated(self, tmp_path: Path) -> None:
        database = tmp_path / "runs.db"
        invoke(CORPUS, database, "--yes")

        assert "already decided and committed" in invoke(CORPUS, database)

    def test_starting_over_is_available(self, tmp_path: Path) -> None:
        database = tmp_path / "runs.db"
        invoke(CORPUS, database, "--yes")

        assert "already decided and committed" not in invoke(CORPUS, database, "--new", "--yes")

    def test_a_document_that_arrived_since_is_named(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        shutil.copytree(CORPUS, corpus)
        database = tmp_path / "runs.db"
        invoke(corpus, database, "--yes")

        (corpus / "AMD-99-later.md").write_text("A later amendment.\n")

        assert "AMD-99-later.md" in invoke(corpus, database)
