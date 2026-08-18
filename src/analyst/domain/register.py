"""Turns reconciled facts into the rows of the deliverable.

Which facts become obligations, who owes them, and how they read is a table
rather than a chain of branches. Adding a term to the register is an entry here;
nothing in the pipeline changes.

A row is only built from a fact that is already grounded, so every row inherits
a span and the register cannot contain an uncited claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from analyst.domain.models import Fact, Obligation, Register
from analyst.domain.reconcile import Reconciliation

# Durations are written either as a numeral or as words with the numeral beside
# them, and both forms appear in these documents.
_SPAN = re.compile(r"([a-z\-]+\s*\(\d+\)|\d+)\s*(?:days|weeks|months)?", re.IGNORECASE)
_PLACE = re.compile(r"(?:the\s+laws?\s+of\s+)?(.+)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class RowSpec:
    """How one subject reads as an obligation."""

    party: str
    duty: str
    """`{value}` is replaced by the fact's value."""

    carries_amount: bool = False

    core: re.Pattern[str] | None = None
    """Pulls the value out of the phrase it was read in.

    The sentence around `{value}` already supplies the words that frame it, so a
    value reported as the whole phrase reads twice: "within within thirty (30)
    days of a written order of a written order". What a model returns here is
    not fixed by the port, so the row is built from the part it needs rather
    than from whatever arrived.
    """

    def core_of(self, value: str) -> str:
        text = value.strip().rstrip(",").strip()
        if self.core is None:
            return text
        match = self.core.search(text)
        return match.group(1).strip() if match else text


ROW_SPECS: dict[str, RowSpec] = {
    "monthly_fee": RowSpec(
        party="Client",
        duty="Pay the monthly fee of {value}, invoiced monthly in arrears",
        carries_amount=True,
    ),
    "payment_terms": RowSpec(
        party="Client",
        duty="Settle invoices within {value} days of the invoice date",
        core=_SPAN,
    ),
    "delivery_window": RowSpec(
        party="Supplier",
        duty="Deliver all materials within {value} days of a written order",
        core=_SPAN,
    ),
    "notice_period": RowSpec(
        party="Either party",
        duty="Give {value} days written notice to terminate",
        core=_SPAN,
    ),
    "data_protection": RowSpec(
        party="Supplier",
        duty="Process personal data only on documented instructions from the Client",
    ),
    "governing_law": RowSpec(
        party="Both parties",
        duty="Be bound by the laws of {value}",
        core=_PLACE,
    ),
}


def build(reconciliation: Reconciliation) -> Register:
    """Assemble the register from the current position.

    Superseded terms are carried as rows marked with the document that replaced
    them, rather than deleted. A register that silently drops what changed
    cannot answer what changed and because of which source.
    """
    rows: list[Obligation] = []

    superseded_by_subject = {
        s.subject: (s.earlier, s.superseding_source_id) for s in reconciliation.supersessions
    }

    for fact in sorted(reconciliation.facts, key=lambda f: f.subject):
        spec = ROW_SPECS.get(fact.subject)
        if spec is None:
            continue

        previous = superseded_by_subject.get(fact.subject)
        if previous is not None:
            earlier, superseding = previous
            rows.append(_row(earlier, ROW_SPECS[earlier.subject]).superseded(superseding))

        rows.append(_row(fact, spec))

    return Register(
        obligations=tuple(rows),
        conflicts=reconciliation.conflicts,
        findings=(),
    )


def _row(fact: Fact, spec: RowSpec) -> Obligation:
    return Obligation(
        obligation_id=f"{fact.subject}:{fact.support[0].source_id}",
        party=spec.party,
        duty=spec.duty.format(value=spec.core_of(fact.value)),
        support=fact.support,
        amount=fact.value if spec.carries_amount else None,
    )


def with_findings(register: Register, findings: tuple[object, ...]) -> Register:
    """Attach findings without disturbing the rows or the conflicts."""
    from analyst.domain.models import Finding

    checked = tuple(f for f in findings if isinstance(f, Finding))
    return Register(
        obligations=register.obligations,
        conflicts=register.conflicts,
        findings=checked,
    )
