"""What one register changed relative to another, and what it did not.

An update has to be able to show that the parts a new document did not affect
stayed exactly as they were. Asserting it is not enough, so every row carries a
digest of its own content and a comparison is made row by row: unchanged rows are
counted and named, not assumed.

The comparison is on content, not on object identity, so a run that produced the
same row twice reports it as unchanged rather than as a replacement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from analyst.domain.models import Conflict, Finding, Obligation, Register


@dataclass(frozen=True, slots=True)
class RegisterDelta:
    """The difference between two registers."""

    added: tuple[Obligation, ...]
    removed: tuple[Obligation, ...]
    changed: tuple[tuple[Obligation, Obligation], ...]
    """Rows present in both, with different content: (before, after)."""

    unchanged: tuple[Obligation, ...]

    new_conflicts: tuple[Conflict, ...]
    resolved_conflicts: tuple[Conflict, ...]
    new_findings: tuple[Finding, ...]
    cleared_findings: tuple[Finding, ...]

    because_of: tuple[str, ...]
    """Source ids cited by everything that changed."""

    @property
    def is_empty(self) -> bool:
        """Whether anything at all moved.

        A re-run over an unchanged corpus lands here, and reporting that plainly
        is the point: an update that changed nothing should say so rather than
        present the same register as though it were new work.
        """
        return not (
            self.added
            or self.removed
            or self.changed
            or self.new_conflicts
            or self.resolved_conflicts
            or self.new_findings
            or self.cleared_findings
        )

    @property
    def untouched_count(self) -> int:
        return len(self.unchanged)

    def summary(self) -> str:
        if self.is_empty:
            return f"nothing changed; {self.untouched_count} rows are byte-identical"
        parts = []
        if self.added:
            parts.append(f"{len(self.added)} added")
        if self.changed:
            parts.append(f"{len(self.changed)} changed")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.new_conflicts:
            parts.append(f"{len(self.new_conflicts)} new conflicts")
        if self.new_findings:
            parts.append(f"{len(self.new_findings)} new findings")
        parts.append(f"{self.untouched_count} untouched")
        return ", ".join(parts)


def compare(before: Register, after: Register) -> RegisterDelta:
    """Diff two registers by row content."""
    before_rows = {row.obligation_id: row for row in before.obligations}
    after_rows = {row.obligation_id: row for row in after.obligations}

    added = tuple(after_rows[i] for i in after_rows.keys() - before_rows.keys())
    removed = tuple(before_rows[i] for i in before_rows.keys() - after_rows.keys())

    changed: list[tuple[Obligation, Obligation]] = []
    unchanged: list[Obligation] = []
    for identifier in before_rows.keys() & after_rows.keys():
        was, now = before_rows[identifier], after_rows[identifier]
        if was.digest() == now.digest():
            unchanged.append(now)
        else:
            changed.append((was, now))

    new_conflicts, resolved_conflicts = _split(before.conflicts, after.conflicts, _conflict_key)
    new_findings, cleared_findings = _split(before.findings, after.findings, _finding_key)

    because_of = {
        span.source_id
        for row in (*added, *removed, *(now for _, now in changed))
        for span in row.support
    }
    because_of |= {span.source_id for c in new_conflicts for span in c.support}
    because_of |= {span.source_id for f in new_findings for span in f.support}

    return RegisterDelta(
        added=added,
        removed=removed,
        changed=tuple(changed),
        unchanged=tuple(unchanged),
        new_conflicts=new_conflicts,
        resolved_conflicts=resolved_conflicts,
        new_findings=new_findings,
        cleared_findings=cleared_findings,
        because_of=tuple(sorted(because_of)),
    )


def _conflict_key(conflict: Conflict) -> str:
    return f"{conflict.subject}|{conflict.left.value}|{conflict.right.value}"


def _finding_key(finding: Finding) -> str:
    return f"{finding.rule_id}|{finding.statement}"


def _split[T](
    before: tuple[T, ...], after: tuple[T, ...], key: Callable[[T], str]
) -> tuple[tuple[T, ...], tuple[T, ...]]:
    """Items new in `after`, and items gone from `before`."""
    before_keys = {key(item) for item in before}
    after_keys = {key(item) for item in after}

    fresh = tuple(item for item in after if key(item) not in before_keys)
    gone = tuple(item for item in before if key(item) not in after_keys)
    return fresh, gone
