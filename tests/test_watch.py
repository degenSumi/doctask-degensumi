"""What the watcher decides to hand over.

A folder is watched by sweeping it, so the two things worth proving are that a
document already read is never offered again, and that a document still being
written is left alone until it stops changing.
"""

from __future__ import annotations

import os
from pathlib import Path

from analyst.app.watch import Arrivals


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    # Timestamps land on the same tick on a fast machine, so age the file to
    # make "changed since the last sweep" mean what it says.
    stamp = os.path.getmtime(path)
    os.utime(path, (stamp - 10, stamp - 10))
    return path


class TestADocumentIsOfferedOnce:
    def test_what_the_run_already_read_never_arrives(self, tmp_path: Path) -> None:
        write(tmp_path, "MSA-2026-014.md", "already read")
        arrivals = Arrivals(known={"MSA-2026-014"})

        assert arrivals.sweep(tmp_path) == []
        assert arrivals.sweep(tmp_path) == []

    def test_a_document_taken_is_not_offered_again(self, tmp_path: Path) -> None:
        path = write(tmp_path, "AMD-02.md", "an amendment")
        arrivals = Arrivals(known=set())

        arrivals.sweep(tmp_path)
        assert arrivals.sweep(tmp_path) == [path]

        arrivals.accept(path.stem)
        assert arrivals.sweep(tmp_path) == []

    def test_a_file_this_system_does_not_read_is_never_offered(self, tmp_path: Path) -> None:
        write(tmp_path, "vendor-billing-checklist.yaml", "rules: []")
        arrivals = Arrivals(known=set())

        arrivals.sweep(tmp_path)
        assert arrivals.sweep(tmp_path) == []


class TestAFileStillArrivingIsLeftAlone:
    def test_it_is_not_handed_over_on_the_sweep_that_finds_it(self, tmp_path: Path) -> None:
        write(tmp_path, "INV-1003.txt", "an invoice")
        arrivals = Arrivals(known=set())

        assert arrivals.sweep(tmp_path) == []

    def test_a_file_that_grew_between_sweeps_waits(self, tmp_path: Path) -> None:
        path = write(tmp_path, "INV-1003.txt", "half")
        arrivals = Arrivals(known=set())

        arrivals.sweep(tmp_path)
        write(tmp_path, "INV-1003.txt", "half and the rest")
        assert arrivals.sweep(tmp_path) == []

        assert arrivals.sweep(tmp_path) == [path]

    def test_a_file_that_vanished_between_sweeps_is_not_an_error(self, tmp_path: Path) -> None:
        path = write(tmp_path, "INV-1003.txt", "an invoice")
        arrivals = Arrivals(known=set())

        arrivals.sweep(tmp_path)
        path.unlink()
        assert arrivals.sweep(tmp_path) == []
