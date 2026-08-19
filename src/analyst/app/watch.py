"""Noticing that a document has arrived.

The folder is polled rather than hooked into an OS notification API. Polling is
one dependency fewer and behaves the same on every machine, and a second's delay
in noticing costs nothing next to a watcher that works on one platform.

A file appearing is not a file that has finished arriving. Anything still
growing is left alone until its size and modification time hold still across two
polls, so a document copied in over a slow link is read once and whole rather
than read truncated and then never read again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from analyst.adapters.loader import discover


@dataclass
class Arrivals:
    """Which documents in a folder the run has not read, and which have settled.

    Holds the source ids the run already knows so a document is offered once,
    and the size and timestamp of each candidate so a partial copy is skipped
    until it stops changing.
    """

    known: set[str]
    marks: dict[Path, tuple[int, float]] = field(default_factory=dict)

    def sweep(self, corpus: Path) -> list[Path]:
        """Documents that are new to the run and unchanged since the last sweep."""
        settled: list[Path] = []
        seen: dict[Path, tuple[int, float]] = {}

        for path in discover(corpus):
            if path.stem in self.known:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue  # removed between listing the folder and reading it
            mark = (stat.st_size, stat.st_mtime)
            seen[path] = mark
            if self.marks.get(path) == mark:
                settled.append(path)

        self.marks = seen
        return settled

    def accept(self, source_id: str) -> None:
        """Record a document as read, so a later sweep does not offer it again."""
        self.known.add(source_id)
