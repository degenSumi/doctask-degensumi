"""Checks sources and the register against the rules a person supplied.

Rules arrive as data. Two kinds exist and they are settled differently:

- rules answerable from one passage, which the model reads and reports on
- rules that only make sense across documents, which the reconciler already
  settled and which are turned into findings here

A rule that is satisfied produces nothing. A corpus where every rule holds
produces an empty tuple, and that is reported as an empty tuple rather than
dressed up as a finding, because "no findings" is a real result.
"""

from __future__ import annotations

from analyst.domain.models import Conflict, Finding, Rule, Severity, Source, SourceKind
from analyst.ports.language_model import RuleVerdict

# Rules the reconciler decides, because they compare two documents. The model is
# never asked about these: it sees one passage and could only guess.
CROSS_DOCUMENT_RULES: dict[str, str] = {
    "TERMS-MATCH-CONTRACT": "payment_terms",
    "FEE-MATCH-CONTRACT": "monthly_fee",
}


def applicable(rules: tuple[Rule, ...], source: Source) -> tuple[Rule, ...]:
    """The rules that apply to one document, per its kind."""
    return tuple(
        rule
        for rule in rules
        if rule.covers(source.kind) and rule.rule_id not in CROSS_DOCUMENT_RULES
    )


def finding_from_verdict(rule: Rule, verdict: RuleVerdict, source: Source) -> Finding | None:
    """Turn a failed check into a finding, or nothing if the rule held.

    A failed check whose quote cannot be located in the source produces nothing.
    A rule cannot be reported as broken without showing where, and a quote that
    is not in the document does not show anything.
    """
    if verdict.satisfied:
        return None

    from analyst.domain.grounding import locate

    span = locate(source, verdict.quote)
    if span is None:
        return None

    return Finding(
        rule_id=rule.rule_id,
        statement=f"{rule.statement.strip()} — {verdict.explanation}",
        severity=rule.severity,
        support=(span,),
    )


def findings_from_conflicts(
    rules: tuple[Rule, ...], conflicts: tuple[Conflict, ...]
) -> tuple[Finding, ...]:
    """Report cross-document rules that the reconciler found broken."""
    by_subject = {subject: rule_id for rule_id, subject in CROSS_DOCUMENT_RULES.items()}
    rules_by_id = {rule.rule_id: rule for rule in rules}

    findings: list[Finding] = []
    for conflict in conflicts:
        rule_id = by_subject.get(conflict.subject)
        rule = rules_by_id.get(rule_id) if rule_id else None
        if rule is None:
            continue

        findings.append(
            Finding(
                rule_id=rule.rule_id,
                statement=f"{rule.statement.strip()} — {conflict.explanation}",
                severity=rule.severity,
                support=conflict.support,
            )
        )

    return tuple(findings)


def load_rules(raw: object) -> tuple[Rule, ...]:
    """Read the rules file into rules, ignoring entries that are not usable.

    A malformed entry is skipped rather than aborting the run, because one bad
    rule should not stop every other check from happening.
    """
    if not isinstance(raw, dict):
        return ()
    entries = raw.get("rules")
    if not isinstance(entries, list):
        return ()

    rules: list[Rule] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rule_id = str(entry.get("id", "")).strip()
        statement = str(entry.get("statement", "")).strip()
        if not rule_id or not statement:
            continue

        try:
            severity = Severity(str(entry.get("severity", "advisory")))
        except ValueError:
            severity = Severity.ADVISORY

        kinds: list[SourceKind] = []
        for name in entry.get("applies_to", []) or []:
            try:
                kinds.append(SourceKind(str(name)))
            except ValueError:
                continue

        rules.append(
            Rule(
                rule_id=rule_id,
                statement=statement,
                severity=severity,
                applies_to=tuple(kinds),
            )
        )

    return tuple(rules)
