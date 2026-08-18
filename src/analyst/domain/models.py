"""Core vocabulary.

Pure: no database, no network, no file parsing. The rules that decide whether a
claim may exist live here, so they are cheap to test and hard to bypass.

The rule this module exists to enforce: **nothing is asserted without a place it
came from**. A fact, a conflict, a finding and a register row each carry at least
one span of source text, and their constructors refuse to build without one.
That is what makes "it never bluffs" a property of the type rather than a
promise in a prompt.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum


class SourceKind(StrEnum):
    """What a document turned out to be.

    Decided by the pipeline rather than by the filename, because a file called
    `invoice_final_v2.pdf` is regularly a contract.
    """

    CONTRACT = "contract"
    AMENDMENT = "amendment"
    INVOICE = "invoice"
    CORRESPONDENCE = "correspondence"
    UNKNOWN = "unknown"
    """Could not be established. Read for facts, but never used to supersede."""


@dataclass(frozen=True, slots=True)
class Source:
    """One document in the pile, as text plus what is known about it."""

    source_id: str
    filename: str
    text: str
    kind: SourceKind = SourceKind.UNKNOWN
    received_at: datetime | None = None
    effective_date: str | None = None
    """Date the document takes effect, used to order amendments."""

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id is required")

    @property
    def digest(self) -> str:
        """Content identity, so re-ingesting the same bytes is free."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def excerpt(self, start: int, end: int) -> str:
        return self.text[start:end]


@dataclass(frozen=True, slots=True)
class Span:
    """An exact stretch of text in one source.

    Offsets are kept alongside the quote so a citation can be verified against
    the source rather than trusted. `verify` is what turns a citation from a
    claim about provenance into a checkable one.
    """

    source_id: str
    start: int
    end: int
    quote: str
    locator: str = ""
    """Human-readable position, such as "clause 4.2" or "line 18"."""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"span {self.start}:{self.end} is not a range")
        if not self.quote.strip():
            raise ValueError("a span must quote the text it points at")

    def verify(self, source: Source) -> bool:
        """Whether this span still says what it claims to say."""
        return (
            source.source_id == self.source_id
            and source.excerpt(self.start, self.end) == self.quote
        )


@dataclass(frozen=True, slots=True)
class Fact:
    """One extracted value, and where it came from."""

    subject: str
    """What the value is about, e.g. "payment_terms" or "total_amount"."""

    value: str
    support: tuple[Span, ...]

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("a fact must say what it is about")
        if not self.support:
            raise ValueError(f"the fact {self.subject!r} has no source span; it cannot be asserted")

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(span.source_id for span in self.support))


@dataclass(frozen=True, slots=True)
class Conflict:
    """Two supported facts about the same subject that cannot both hold.

    Surfaced rather than resolved. Which one is right is a judgement about the
    business, and the system does not have the standing to make it.
    """

    subject: str
    left: Fact
    right: Fact
    explanation: str

    def __post_init__(self) -> None:
        if self.left.subject != self.right.subject:
            raise ValueError("a conflict must be between facts about the same subject")
        if self.left.value == self.right.value:
            raise ValueError("facts that agree are not a conflict")

    @property
    def support(self) -> tuple[Span, ...]:
        return (*self.left.support, *self.right.support)


class Severity(StrEnum):
    BLOCKING = "blocking"
    ADVISORY = "advisory"


@dataclass(frozen=True, slots=True)
class Rule:
    """One check the user asked for, as data rather than as code."""

    rule_id: str
    statement: str
    severity: Severity = Severity.ADVISORY
    applies_to: tuple[SourceKind, ...] = ()
    """Empty means every kind."""

    def covers(self, kind: SourceKind) -> bool:
        return not self.applies_to or kind in self.applies_to


@dataclass(frozen=True, slots=True)
class Finding:
    """A rule that was not satisfied, and the text that shows it."""

    rule_id: str
    statement: str
    severity: Severity
    support: tuple[Span, ...]

    def __post_init__(self) -> None:
        if not self.support:
            raise ValueError(
                f"the finding for {self.rule_id!r} has no source span; a rule cannot be "
                "reported as failed without showing where"
            )


@dataclass(frozen=True, slots=True)
class Obligation:
    """One row of the deliverable: who must do what, by when, under what."""

    obligation_id: str
    party: str
    duty: str
    support: tuple[Span, ...]
    due: str | None = None
    amount: str | None = None
    superseded_by: str | None = None
    """Source id of the amendment that replaced this, if any."""

    def __post_init__(self) -> None:
        if not self.support:
            raise ValueError(
                f"the obligation {self.obligation_id!r} has no source span; every row of "
                "the register must trace to the document it came from"
            )

    @property
    def is_current(self) -> bool:
        return self.superseded_by is None

    def superseded(self, by_source_id: str) -> Obligation:
        return replace(self, superseded_by=by_source_id)

    def digest(self) -> str:
        """Identity of the row's content, used to tell a real change from a rerun."""
        material = "␟".join(
            (self.party, self.duty, self.due or "", self.amount or "", self.superseded_by or "")
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Register:
    """The deliverable: current obligations, open conflicts, findings."""

    obligations: tuple[Obligation, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    findings: tuple[Finding, ...] = ()

    @property
    def current(self) -> tuple[Obligation, ...]:
        return tuple(o for o in self.obligations if o.is_current)

    def digest(self) -> str:
        """Content identity of the whole register.

        Two runs over the same corpus produce the same digest. An update that
        changed nothing is therefore detectable rather than assumed.
        """
        material = "␟".join(
            sorted(o.digest() for o in self.obligations)
            + sorted(c.subject + c.left.value + c.right.value for c in self.conflicts)
            + sorted(f.rule_id + f.statement for f in self.findings)
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def rows_touching(self, source_id: str) -> tuple[Obligation, ...]:
        return tuple(
            o for o in self.obligations if any(s.source_id == source_id for s in o.support)
        )


class ProposalKind(StrEnum):
    """What a pending change would do to the register if approved."""

    ADD_OBLIGATION = "add_obligation"
    SUPERSEDE_OBLIGATION = "supersede_obligation"
    CONFLICT = "conflict"
    FINDING = "finding"
    QUARANTINE = "quarantine"
    """A source containing text aimed at the system. Reported, never followed."""


class Decision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class Proposal:
    """One reviewable item. Nothing reaches the register without a decision."""

    proposal_id: str
    kind: ProposalKind
    summary: str
    support: tuple[Span, ...]
    reason: str = ""
    decided_by: str = ""
    obligation: Obligation | None = None
    conflict: Conflict | None = None
    finding: Finding | None = None
    target_obligation_id: str | None = None
    decision: Decision | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.support:
            raise ValueError(
                f"the proposal {self.proposal_id!r} has no source span; a person cannot "
                "review a change that does not say where it came from"
            )
        if self.kind is ProposalKind.ADD_OBLIGATION and self.obligation is None:
            raise ValueError("an add proposal must carry the obligation it would add")
        if self.kind is ProposalKind.SUPERSEDE_OBLIGATION and not self.target_obligation_id:
            raise ValueError("a supersede proposal must name the row it would replace")
        if self.kind is ProposalKind.CONFLICT and self.conflict is None:
            raise ValueError("a conflict proposal must carry the conflict")
        if self.kind is ProposalKind.FINDING and self.finding is None:
            raise ValueError("a finding proposal must carry the finding")

    @property
    def is_decided(self) -> bool:
        return self.decision is not None

    @property
    def changes_the_register(self) -> bool:
        """Whether approving this would alter a row.

        Conflicts, findings and quarantines are reported without changing the
        register, so approving one records that a person saw it rather than
        silently editing the deliverable.
        """
        return self.kind in (ProposalKind.ADD_OBLIGATION, ProposalKind.SUPERSEDE_OBLIGATION)

    def with_decision(self, decision: Decision) -> Proposal:
        return replace(self, decision=decision)
