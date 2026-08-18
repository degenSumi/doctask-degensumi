"""Converts domain objects to and from the plain data a checkpoint holds.

Graph state is written to a checkpointer after every stage and read back on
resume, so it has to be plain JSON-compatible data. Conversion is written out by
hand rather than derived, because the stored shape is what a resumed run depends
on: it is stated explicitly here and covered by a round-trip test.
"""

from __future__ import annotations

from typing import Any

from analyst.domain.models import (
    Conflict,
    Decision,
    Fact,
    Finding,
    Obligation,
    Proposal,
    ProposalKind,
    Register,
    Severity,
    Source,
    SourceKind,
    Span,
)


def span_to(span: Span) -> dict[str, Any]:
    return {
        "source_id": span.source_id,
        "start": span.start,
        "end": span.end,
        "quote": span.quote,
        "locator": span.locator,
    }


def span_from(raw: dict[str, Any]) -> Span:
    return Span(**raw)


def spans_to(spans: tuple[Span, ...]) -> list[dict[str, Any]]:
    return [span_to(s) for s in spans]


def spans_from(raw: list[dict[str, Any]]) -> tuple[Span, ...]:
    return tuple(span_from(s) for s in raw)


def source_to(source: Source) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "filename": source.filename,
        "text": source.text,
        "kind": str(source.kind),
        "received_at": source.received_at.isoformat() if source.received_at else None,
        "effective_date": source.effective_date,
    }


def source_from(raw: dict[str, Any]) -> Source:
    from datetime import datetime

    return Source(
        source_id=raw["source_id"],
        filename=raw["filename"],
        text=raw["text"],
        kind=SourceKind(raw["kind"]),
        received_at=datetime.fromisoformat(raw["received_at"]) if raw["received_at"] else None,
        effective_date=raw["effective_date"],
    )


def fact_to(fact: Fact) -> dict[str, Any]:
    return {"subject": fact.subject, "value": fact.value, "support": spans_to(fact.support)}


def fact_from(raw: dict[str, Any]) -> Fact:
    return Fact(subject=raw["subject"], value=raw["value"], support=spans_from(raw["support"]))


def conflict_to(conflict: Conflict) -> dict[str, Any]:
    return {
        "subject": conflict.subject,
        "left": fact_to(conflict.left),
        "right": fact_to(conflict.right),
        "explanation": conflict.explanation,
    }


def conflict_from(raw: dict[str, Any]) -> Conflict:
    return Conflict(
        subject=raw["subject"],
        left=fact_from(raw["left"]),
        right=fact_from(raw["right"]),
        explanation=raw["explanation"],
    )


def finding_to(finding: Finding) -> dict[str, Any]:
    return {
        "rule_id": finding.rule_id,
        "statement": finding.statement,
        "severity": str(finding.severity),
        "support": spans_to(finding.support),
    }


def finding_from(raw: dict[str, Any]) -> Finding:
    return Finding(
        rule_id=raw["rule_id"],
        statement=raw["statement"],
        severity=Severity(raw["severity"]),
        support=spans_from(raw["support"]),
    )


def obligation_to(obligation: Obligation) -> dict[str, Any]:
    return {
        "obligation_id": obligation.obligation_id,
        "party": obligation.party,
        "duty": obligation.duty,
        "support": spans_to(obligation.support),
        "due": obligation.due,
        "amount": obligation.amount,
        "superseded_by": obligation.superseded_by,
    }


def obligation_from(raw: dict[str, Any]) -> Obligation:
    return Obligation(
        obligation_id=raw["obligation_id"],
        party=raw["party"],
        duty=raw["duty"],
        support=spans_from(raw["support"]),
        due=raw["due"],
        amount=raw["amount"],
        superseded_by=raw["superseded_by"],
    )


def proposal_to(proposal: Proposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "kind": str(proposal.kind),
        "summary": proposal.summary,
        "support": spans_to(proposal.support),
        "reason": proposal.reason,
        "decided_by": proposal.decided_by,
        "obligation": obligation_to(proposal.obligation) if proposal.obligation else None,
        "conflict": conflict_to(proposal.conflict) if proposal.conflict else None,
        "finding": finding_to(proposal.finding) if proposal.finding else None,
        "target_obligation_id": proposal.target_obligation_id,
        "decision": str(proposal.decision) if proposal.decision else None,
        "notes": list(proposal.notes),
    }


def proposal_from(raw: dict[str, Any]) -> Proposal:
    return Proposal(
        proposal_id=raw["proposal_id"],
        kind=ProposalKind(raw["kind"]),
        summary=raw["summary"],
        support=spans_from(raw["support"]),
        reason=raw["reason"],
        decided_by=raw["decided_by"],
        obligation=obligation_from(raw["obligation"]) if raw["obligation"] else None,
        conflict=conflict_from(raw["conflict"]) if raw["conflict"] else None,
        finding=finding_from(raw["finding"]) if raw["finding"] else None,
        target_obligation_id=raw["target_obligation_id"],
        decision=Decision(raw["decision"]) if raw["decision"] else None,
        notes=tuple(raw.get("notes", [])),
    )


def register_to(register: Register) -> dict[str, Any]:
    return {
        "obligations": [obligation_to(o) for o in register.obligations],
        "conflicts": [conflict_to(c) for c in register.conflicts],
        "findings": [finding_to(f) for f in register.findings],
    }


def register_from(raw: dict[str, Any]) -> Register:
    return Register(
        obligations=tuple(obligation_from(o) for o in raw.get("obligations", [])),
        conflicts=tuple(conflict_from(c) for c in raw.get("conflicts", [])),
        findings=tuple(finding_from(f) for f in raw.get("findings", [])),
    )
