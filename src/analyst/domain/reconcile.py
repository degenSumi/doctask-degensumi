"""Decides what the pile collectively says, and where it does not agree.

The model reads one passage at a time and reports what it says. Nothing it
returns is a judgement about the pile as a whole. That judgement is made here,
in code, over facts that are already grounded in text.

The distinction that does the work: an amendment changing a contract term is a
**supersession**, which is the documents working as intended. An invoice
disagreeing with the term in force on its date is a **conflict**, which is a
person's problem. Treating the first as a conflict would bury the second in
noise, and treating the second as a supersession would silently accept a
mis-billed invoice.

Nothing here resolves a conflict. Which side is right is a question about the
business, and the system has no standing to answer it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime

from analyst.domain.models import Conflict, Fact, Source, SourceKind

# Subjects on an invoice, and the agreement subject each must match.
INVOICE_COMPLIANCE: dict[str, str] = {
    "invoice_terms": "payment_terms",
    "invoice_total": "monthly_fee",
}

_DATE_FORMATS = ("%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%d/%m/%Y")


def parse_date(value: str | None) -> date | None:
    """Read a date written the way these documents write them.

    Returns nothing rather than guessing on an unrecognised format, so an
    unreadable date cannot silently order an amendment wrongly.
    """
    if not value:
        return None
    cleaned = value.strip().rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


@dataclass(frozen=True, slots=True)
class Supersession:
    """A later document replacing an earlier document's term."""

    subject: str
    earlier: Fact
    later: Fact
    superseded_source_id: str
    superseding_source_id: str
    effective_from: date | None


@dataclass(frozen=True, slots=True)
class Reconciliation:
    facts: tuple[Fact, ...]
    """One current fact per subject, after supersession."""

    supersessions: tuple[Supersession, ...]
    conflicts: tuple[Conflict, ...]

    @property
    def by_subject(self) -> dict[str, Fact]:
        return {fact.subject: fact for fact in self.facts}


def reconcile(facts: tuple[Fact, ...], sources: dict[str, Source]) -> Reconciliation:
    """Work out the current position, and where the documents disagree."""
    agreement_facts = [f for f in facts if _kind_of(f, sources) in _AGREEMENT_KINDS]
    invoice_facts = [f for f in facts if _kind_of(f, sources) is SourceKind.INVOICE]

    timeline, supersessions, agreement_conflicts = _settle_agreement(agreement_facts, sources)
    invoice_dates = _invoice_dates(facts, sources)
    invoice_conflicts = _check_invoices(invoice_facts, timeline, invoice_dates)

    current = {subject: entries[-1][1] for subject, entries in timeline.items()}

    return Reconciliation(
        facts=tuple(current.values()),
        supersessions=tuple(supersessions),
        conflicts=tuple([*agreement_conflicts, *invoice_conflicts]),
    )


def _invoice_dates(facts: tuple[Fact, ...], sources: dict[str, Source]) -> dict[str, date | None]:
    """The date each invoice was issued, read from the invoice itself."""
    dates: dict[str, date | None] = {}
    for fact in facts:
        if fact.subject != "invoice_date":
            continue
        source_id = fact.support[0].source_id
        if sources.get(source_id) and sources[source_id].kind is SourceKind.INVOICE:
            dates[source_id] = parse_date(fact.value)
    return dates


def in_force_on(
    timeline: dict[str, list[tuple[date | None, Fact]]], subject: str, when: date | None
) -> Fact | None:
    """The term that applied on a given date.

    An invoice is measured against the agreement as it stood when the invoice
    was issued, not as it stands now. Comparing against the latest terms would
    report every invoice raised before an amendment as wrong.

    When the invoice carries no readable date, nothing is returned: the
    comparison is skipped and reported as undecidable rather than made against
    an assumed date.
    """
    entries = timeline.get(subject)
    if not entries:
        return None
    if when is None:
        return None

    applicable = [fact for effective, fact in entries if effective is None or effective <= when]
    return applicable[-1] if applicable else None


_AGREEMENT_KINDS = (SourceKind.CONTRACT, SourceKind.AMENDMENT)


def _kind_of(fact: Fact, sources: dict[str, Source]) -> SourceKind:
    first = fact.support[0].source_id
    source = sources.get(first)
    return source.kind if source else SourceKind.UNKNOWN


def _ordering_key(fact: Fact, sources: dict[str, Source]) -> tuple[int, str, str]:
    """Order agreement facts so a term is settled before anything amends it.

    Contracts come first, then amendments by effective date. An amendment whose
    date could not be read sorts last rather than being dropped: it still gets
    seen, just after every dated one, so an unreadable date degrades the
    ordering instead of losing the document.
    """
    source_id = fact.support[0].source_id
    source = sources.get(source_id)
    kind = source.kind if source else SourceKind.UNKNOWN

    rank = 0 if kind is SourceKind.CONTRACT else 1
    effective = parse_date(source.effective_date if source else None)
    stamp = effective.isoformat() if effective else "9999-12-31"

    return (rank, stamp, source_id)


def _settle_agreement(
    facts: list[Fact], sources: dict[str, Source]
) -> tuple[dict[str, list[tuple[date | None, Fact]]], list[Supersession], list[Conflict]]:
    """Build the history of each term, in effective-date order.

    A history rather than a single value, because an invoice has to be measured
    against the term that applied when it was issued.
    """
    timeline: dict[str, list[tuple[date | None, Fact]]] = {}
    current: dict[str, Fact] = {}
    supersessions: list[Supersession] = []
    conflicts: list[Conflict] = []

    for fact in sorted(facts, key=lambda f: _ordering_key(f, sources)):
        source_id = fact.support[0].source_id
        effective = parse_date(sources[source_id].effective_date if source_id in sources else None)

        existing = current.get(fact.subject)
        if existing is None:
            current[fact.subject] = fact
            timeline.setdefault(fact.subject, []).append((effective, fact))
            continue

        if existing.value == fact.value:
            continue

        if _kind_of(fact, sources) is SourceKind.AMENDMENT:
            supersessions.append(
                Supersession(
                    subject=fact.subject,
                    earlier=existing,
                    later=fact,
                    superseded_source_id=existing.support[0].source_id,
                    superseding_source_id=source_id,
                    effective_from=effective,
                )
            )
            current[fact.subject] = fact
            timeline.setdefault(fact.subject, []).append((effective, fact))
            continue

        # Two agreements of the same standing disagreeing is not something a
        # later date resolves. It goes to a person.
        conflicts.append(
            Conflict(
                subject=fact.subject,
                left=existing,
                right=fact,
                explanation=(
                    f"Two agreement documents state different values for "
                    f"{fact.subject.replace('_', ' ')}, and neither amends the other."
                ),
            )
        )

    return timeline, supersessions, conflicts


def _check_invoices(
    invoice_facts: list[Fact],
    timeline: dict[str, list[tuple[date | None, Fact]]],
    invoice_dates: dict[str, date | None],
) -> list[Conflict]:
    """Compare each invoice against the term in force when it was issued."""
    conflicts: list[Conflict] = []

    for fact in invoice_facts:
        agreement_subject = INVOICE_COMPLIANCE.get(fact.subject)
        if agreement_subject is None:
            continue

        issued = invoice_dates.get(fact.support[0].source_id)
        in_force = in_force_on(timeline, agreement_subject, issued)
        if in_force is None:
            continue

        if _values_agree(fact.value, in_force.value):
            continue

        # The invoice states the same term under its own label, so it is
        # relabelled to the agreement's subject before the two are compared as
        # one disagreement. Its spans are untouched, so the citation still
        # points at the invoice.
        stated = replace(fact, subject=agreement_subject)

        conflicts.append(
            Conflict(
                subject=agreement_subject,
                left=in_force,
                right=stated,
                explanation=(
                    f"The agreement states {in_force.value!r} for "
                    f"{agreement_subject.replace('_', ' ')}, and the invoice states "
                    f"{fact.value!r}."
                ),
            )
        )

    return conflicts


def _values_agree(left: str, right: str) -> bool:
    """Whether two written values mean the same thing.

    Compares the numbers in each, because "net 45", "net forty-five (45) days"
    and "45 days" are the same term written three ways, while a difference in
    the digits is a real difference.
    """
    left_numbers = _numbers(left)
    right_numbers = _numbers(right)
    if left_numbers and right_numbers:
        return bool(left_numbers & right_numbers)
    return _normalise(left) == _normalise(right)


def _numbers(value: str) -> set[str]:
    import re

    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*", value)}


def _normalise(value: str) -> str:
    return " ".join(value.lower().split())
