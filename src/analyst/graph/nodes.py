"""The stages, and the decisions that change which stage runs next.

Three decisions can move a run off the straight path, and each is recorded in
the stage log by name:

- **skip** — a document that yields nothing usable is set aside with a reason,
  rather than failing the run or being silently counted as read
- **retry** — extraction that finds nothing in the opening passage is tried once
  more against the whole document, because the values are often further down
- **escalate** — a document containing text aimed at the system is quarantined
  and routed to a person, never to the extractor as an instruction

Nothing here writes to the register. Stages produce proposals; a person decides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from analyst.adapters.loader import UnsupportedFormat, discover, read_text
from analyst.domain.examine import (
    applicable,
    finding_from_verdict,
    findings_from_conflicts,
    load_rules,
)
from analyst.domain.grounding import ground
from analyst.domain.models import (
    Proposal,
    ProposalKind,
    Register,
    Source,
    SourceKind,
    Span,
)
from analyst.domain.reconcile import reconcile
from analyst.domain.register import build, with_findings
from analyst.domain.rules import detect_instructions, redact
from analyst.graph.serde import (
    conflict_to,
    fact_from,
    fact_to,
    proposal_to,
    register_to,
    source_from,
    source_to,
)
from analyst.graph.state import RunState, log_entry
from analyst.ports.language_model import LanguageModel, ModelUnavailable

SUBJECTS: dict[str, str] = {
    # Several of these are near neighbours in a document, so each says what it
    # means. Asked for bare names, a reader picks the one whose words match the
    # page rather than the one that was meant.
    "payment_terms": "the payment terms an agreement or amendment sets",
    "invoice_terms": "the payment terms printed on an invoice itself",
    "monthly_fee": "the recurring fee an agreement or amendment sets",
    "invoice_total": "the total amount an invoice asks for",
    "invoice_date": "the date an invoice was issued",
    "invoice_number": "the reference an invoice gives itself",
    "purchase_order": "the purchase order number an invoice quotes",
    "governing_law": "the jurisdiction whose law governs the agreement",
    "delivery_window": "how long the supplier has to deliver after an order",
    "notice_period": "how much notice is needed to terminate",
    "data_protection": "the duty an agreement places on handling personal data",
    "agreement_reference": "the agreement a document says it belongs to",
}

HEAD_CHARS = 4000
"""How much of a document the first extraction pass reads."""


class Nodes:
    """Stage functions bound to one model.

    A class rather than free functions so the model is injected once and every
    stage uses the same one, which is what makes the cost report a total for the
    run rather than a guess.
    """

    def __init__(self, model: LanguageModel) -> None:
        self._model = model
        self._counted = self._model.cost

    def _spent(self) -> dict[str, int]:
        """What the model consumed since this was last asked.

        Reported as a delta rather than a total, because the state reducer sums
        it. A resumed run therefore keeps what the first attempt spent: that
        work was paid for whether or not the process survived to use it.
        """
        now = self._model.cost
        delta = {
            "calls": now.calls - self._counted.calls,
            "input_chars": now.input_tokens - self._counted.input_tokens,
            "output_chars": now.output_tokens - self._counted.output_tokens,
        }
        self._counted = now
        return delta

    # -- ingest -------------------------------------------------------------

    def ingest(self, state: RunState) -> dict[str, Any]:
        """Read every readable document in the corpus folder."""
        folder = Path(state["corpus_dir"])
        sources: dict[str, Any] = dict(state.get("sources") or {})
        log: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        for path in discover(folder):
            source_id = path.stem
            if source_id in sources:
                # Already read on an earlier attempt. Re-reading would spend the
                # work again for the same bytes.
                continue
            try:
                text = read_text(path)
            except (UnsupportedFormat, OSError) as error:
                skipped.append({"source": source_id, "reason": str(error)})
                log.append(log_entry("ingest", "skip", str(error), source=source_id))
                continue

            sources[source_id] = source_to(
                Source(source_id=source_id, filename=path.name, text=text)
            )
            log.append(log_entry("ingest", "read", f"{len(text)} characters", source=source_id))

        log.append(log_entry("ingest", "done", f"{len(sources)} sources"))
        return {"sources": sources, "stage_log": log, "skipped": skipped}

    # -- classify -----------------------------------------------------------

    def classify(self, state: RunState) -> dict[str, Any]:
        """Decide what each document is, and quarantine any that give orders."""
        sources: dict[str, Any] = dict(state.get("sources") or {})
        log: list[dict[str, Any]] = []
        quarantined: list[str] = list(state.get("quarantined") or [])

        for source_id, raw in sources.items():
            source = source_from(raw)
            if source.kind is not SourceKind.UNKNOWN:
                # Already classified on an earlier pass. An arrival should cost
                # what the arrival costs, not a pass over everything read so far.
                continue

            hostile = detect_instructions(source.text)
            if hostile and source_id not in quarantined:
                quarantined.append(source_id)
                names = sorted({d.pattern.name for d in hostile})
                log.append(
                    log_entry(
                        "classify",
                        "escalate",
                        f"contains text addressed to the system ({', '.join(names)})",
                        source=source_id,
                    )
                )

            try:
                verdict = self._model.classify_source(source.filename, source.text[:HEAD_CHARS])
            except ModelUnavailable as error:
                log.append(log_entry("classify", "skip", str(error), source=source_id))
                continue

            sources[source_id] = source_to(
                Source(
                    source_id=source.source_id,
                    filename=source.filename,
                    text=source.text,
                    kind=verdict.kind,
                    effective_date=verdict.effective_date,
                )
            )
            log.append(
                log_entry(
                    "classify",
                    "classified",
                    f"{verdict.kind}"
                    + (f", effective {verdict.effective_date}" if verdict.effective_date else ""),
                    source=source_id,
                )
            )

        return {
            "sources": sources,
            "quarantined": quarantined,
            "stage_log": log,
            "cost": self._spent(),
        }

    # -- extract ------------------------------------------------------------

    def extract(self, state: RunState) -> dict[str, Any]:
        """Pull grounded facts out of each document.

        A quarantined document is read with its hostile passages removed, so the
        ordinary content around them still contributes rather than the whole
        document being discarded because of one sentence.
        """
        sources = state.get("sources") or {}
        quarantined = set(state.get("quarantined") or [])
        retries: dict[str, Any] = dict(state.get("retries") or {})

        facts: list[dict[str, Any]] = list(state.get("facts") or [])
        already = {(f["subject"], f["support"][0]["source_id"]) for f in facts}
        extracted: list[str] = list(state.get("extracted") or [])
        log: list[dict[str, Any]] = []

        for source_id, raw in sources.items():
            if source_id in extracted:
                # Read on an earlier pass. Asking again would spend a call to be
                # told the same thing and then discard it as a duplicate.
                continue

            source = source_from(raw)
            readable = source
            if source_id in quarantined:
                cleaned = redact(source.text, detect_instructions(source.text))
                readable = Source(
                    source_id=source.source_id,
                    filename=source.filename,
                    text=cleaned,
                    kind=source.kind,
                    effective_date=source.effective_date,
                )

            attempt = int(retries.get(source_id, 0))
            if attempt >= 2:
                # Already read in full on the retry pass. Reading it a third
                # time would spend the work for the same answer.
                continue

            passage = readable.text if attempt else readable.text[:HEAD_CHARS]
            if attempt == 1:
                # Mark the retry as spent before doing it, so the router cannot
                # send this document round again whatever the outcome.
                retries[source_id] = 2

            try:
                found = self._model.extract(SUBJECTS, passage)
            except ModelUnavailable as error:
                log.append(log_entry("extract", "skip", str(error), source=source_id))
                continue

            grounded = 0
            for extraction in found:
                if (extraction.subject, source_id) in already:
                    continue
                fact = ground(readable, extraction.subject, extraction.value, extraction.quote)
                if fact is None:
                    # The quote is not in the document. Dropped rather than
                    # recorded, which is what stops a fluent guess reaching the
                    # register.
                    log.append(
                        log_entry(
                            "extract",
                            "discard",
                            f"{extraction.subject}: quoted text is not in the document",
                            source=source_id,
                        )
                    )
                    continue
                facts.append(fact_to(fact))
                already.add((extraction.subject, source_id))
                grounded += 1

            if grounded == 0 and attempt == 0 and len(readable.text) > HEAD_CHARS:
                # The opening passage carried nothing. Read the whole document
                # once before concluding there is nothing in it.
                retries[source_id] = 1
                log.append(
                    log_entry(
                        "extract",
                        "retry",
                        "no facts in the opening passage; reading the whole document",
                        source=source_id,
                    )
                )
                continue

            extracted.append(source_id)
            log.append(
                log_entry("extract", "extracted", f"{grounded} grounded facts", source=source_id)
            )

        return {
            "facts": facts,
            "retries": retries,
            "extracted": extracted,
            "stage_log": log,
            "cost": self._spent(),
        }

    # -- reconcile ----------------------------------------------------------

    def reconcile(self, state: RunState) -> dict[str, Any]:
        """Work out the current position and where the documents disagree."""
        sources = {sid: source_from(raw) for sid, raw in (state.get("sources") or {}).items()}
        facts = tuple(fact_from(f) for f in state.get("facts") or [])

        result = reconcile(facts, sources)

        log = [
            log_entry(
                "reconcile",
                "settled",
                f"{len(result.facts)} current terms, "
                f"{len(result.supersessions)} superseded, "
                f"{len(result.conflicts)} conflicts",
            )
        ]
        for supersession in result.supersessions:
            log.append(
                log_entry(
                    "reconcile",
                    "supersede",
                    f"{supersession.subject}: {supersession.earlier.value!r} -> "
                    f"{supersession.later.value!r}",
                    source=supersession.superseding_source_id,
                )
            )
        for conflict in result.conflicts:
            log.append(log_entry("reconcile", "conflict", conflict.explanation))

        register = build(result)

        return {
            "conflicts": [conflict_to(c) for c in result.conflicts],
            "supersessions": [
                {
                    "subject": s.subject,
                    "from": s.earlier.value,
                    "to": s.later.value,
                    "by": s.superseding_source_id,
                    "effective": s.effective_from.isoformat() if s.effective_from else None,
                }
                for s in result.supersessions
            ],
            "register": register_to(register),
            "stage_log": log,
        }

    # -- examine ------------------------------------------------------------

    def examine(self, state: RunState) -> dict[str, Any]:
        """Check the sources and the register against the supplied rules."""
        from analyst.graph.serde import conflict_from, finding_from, finding_to, register_from

        rules_path = Path(state.get("rules_path") or "")
        if not rules_path.exists():
            return {"stage_log": [log_entry("examine", "skip", f"no rules file at {rules_path}")]}

        rules = load_rules(yaml.safe_load(rules_path.read_text(encoding="utf-8")))
        sources = {sid: source_from(raw) for sid, raw in (state.get("sources") or {}).items()}
        conflicts = tuple(conflict_from(c) for c in state.get("conflicts") or [])

        # Cross-document rules are settled from the conflicts, which the
        # reconciler has just recomputed over the whole pile. They are cheap and
        # always current.
        findings = list(findings_from_conflicts(rules, conflicts))

        # Single-document rules are checked once per source. On an update only
        # the newly arrived documents are checked, so an update costs what the
        # arrival costs rather than what a full pass costs.
        #
        # Their findings are carried in state rather than read back out of the
        # register, because reconcile rebuilds the register from facts on every
        # pass and a finding about a document nobody re-examined would be lost.
        examined: list[str] = list(state.get("examined") or [])
        kept: list[dict[str, Any]] = list(state.get("findings") or [])
        findings.extend(finding_from(f) for f in kept)

        log: list[dict[str, Any]] = [
            log_entry("examine", "loaded", f"{len(rules)} rules from {rules_path.name}")
        ]

        # Cross-document findings are recorded by name too. Logging only the
        # rule-checked ones left the stage reporting fewer findings than the
        # register ended up carrying.
        for cross_document in findings:
            log.append(
                log_entry(
                    "examine",
                    "finding",
                    f"{cross_document.rule_id}: {cross_document.statement}",
                    source=cross_document.support[0].source_id,
                )
            )

        for source in sources.values():
            if source.kind is SourceKind.UNKNOWN or source.source_id in examined:
                continue
            examined.append(source.source_id)
            for rule in applicable(rules, source):
                try:
                    verdict = self._model.check_rule(
                        rule.rule_id, rule.statement, source.text[:HEAD_CHARS]
                    )
                except ModelUnavailable as error:
                    log.append(log_entry("examine", "skip", str(error), source=source.source_id))
                    continue

                finding = finding_from_verdict(rule, verdict, source)
                if finding is not None:
                    findings.append(finding)
                    kept.append(finding_to(finding))
                    log.append(
                        log_entry(
                            "examine",
                            "finding",
                            f"{rule.rule_id}: {verdict.explanation}",
                            source=source.source_id,
                        )
                    )

        if not findings:
            log.append(log_entry("examine", "clean", "every rule held; no findings"))

        deduped = {(f.rule_id, f.statement): f for f in findings}
        register = with_findings(
            register_from(state.get("register") or {}), tuple(deduped.values())
        )
        return {
            "register": register_to(register),
            "examined": examined,
            "findings": kept,
            "stage_log": log,
            "cost": self._spent(),
        }

    # -- propose ------------------------------------------------------------

    def propose(self, state: RunState) -> dict[str, Any]:
        """Turn everything the run produced into items a person decides on."""
        from analyst.domain.models import Decision
        from analyst.graph.serde import register_from

        register: Register = register_from(state.get("register") or {})
        sources = {sid: source_from(raw) for sid, raw in (state.get("sources") or {}).items()}
        quarantined = state.get("quarantined") or []

        # A decision already taken on an item that has not moved is carried
        # forward. An arrival that touched two rows should not put the other
        # fourteen back in front of a person: that is the cost this path exists
        # to avoid, and re-deciding them is where a tired reviewer waves through
        # the one thing that did change.
        previous = {p["proposal_id"]: p for p in (state.get("proposals") or [])}

        def settled(proposal: Proposal) -> Proposal:
            was = previous.get(proposal.proposal_id)
            if was is None or not was.get("decision"):
                return proposal

            # Matched on content, not on id, so a row that changed under the
            # same id is asked about again.
            def body(item: dict[str, Any]) -> dict[str, Any]:
                return {k: v for k, v in item.items() if k not in ("decision", "notes")}

            if body(was) != body(proposal_to(proposal)):
                return proposal
            return proposal.with_decision(Decision(was["decision"]))

        proposals: list[Proposal] = []

        for row in register.obligations:
            proposals.append(
                settled(
                    Proposal(
                        proposal_id=f"row:{row.obligation_id}",
                        kind=(
                            ProposalKind.SUPERSEDE_OBLIGATION
                            if row.superseded_by
                            else ProposalKind.ADD_OBLIGATION
                        ),
                        summary=f"{row.party}: {row.duty}",
                        support=row.support,
                        reason=(
                            f"superseded by {row.superseded_by}"
                            if row.superseded_by
                            else "current term"
                        ),
                        decided_by="reconcile",
                        obligation=row,
                        target_obligation_id=row.obligation_id if row.superseded_by else None,
                    )
                )
            )

        for index, conflict in enumerate(register.conflicts):
            proposals.append(
                settled(
                    Proposal(
                        proposal_id=f"conflict:{index}",
                        kind=ProposalKind.CONFLICT,
                        summary=f"{conflict.subject}: the documents disagree",
                        support=conflict.support,
                        reason=conflict.explanation,
                        decided_by="reconcile",
                        conflict=conflict,
                    )
                )
            )

        for index, finding in enumerate(register.findings):
            proposals.append(
                settled(
                    Proposal(
                        proposal_id=f"finding:{index}:{finding.rule_id}",
                        kind=ProposalKind.FINDING,
                        summary=f"{finding.rule_id} not satisfied",
                        support=finding.support,
                        reason=finding.statement,
                        decided_by="examine",
                        finding=finding,
                    )
                )
            )

        for source_id in quarantined:
            source = sources.get(source_id)
            if source is None:
                continue
            detections = detect_instructions(source.text)
            if not detections:
                continue
            first = detections[0]
            proposals.append(
                settled(
                    Proposal(
                        proposal_id=f"quarantine:{source_id}",
                        kind=ProposalKind.QUARANTINE,
                        summary=f"{source.filename} contains text addressed to the system",
                        support=(
                            Span(
                                source_id=source_id,
                                start=first.start,
                                end=first.end,
                                quote=first.quote,
                                locator="quarantined passage",
                            ),
                        ),
                        reason=(
                            f"{len(detections)} passage(s) matched: "
                            + ", ".join(sorted({d.pattern.name for d in detections}))
                            + ". Reported as a finding; not acted on."
                        ),
                        decided_by="rule:instruction-detection",
                    )
                )
            )

        waiting = sum(1 for p in proposals if not p.is_decided)
        detail = f"{waiting} items awaiting a decision"
        if waiting != len(proposals):
            detail += f", {len(proposals) - waiting} carried forward unchanged"
        log = [log_entry("propose", "prepared", detail)]
        return {"proposals": [proposal_to(p) for p in proposals], "stage_log": log}

    # -- commit -------------------------------------------------------------

    def commit(self, state: RunState) -> dict[str, Any]:
        """Apply the decisions a person made, and nothing else."""
        from analyst.graph.serde import proposal_from, register_from

        proposals = [proposal_from(p) for p in state.get("proposals") or []]
        register = register_from(state.get("register") or {})

        undecided = [p.proposal_id for p in proposals if not p.is_decided]
        if undecided:
            raise ValueError(
                f"{len(undecided)} item(s) have no decision: {undecided[:5]}. "
                "Nothing is written until every item has been decided."
            )

        from analyst.domain.models import Decision

        approved_rows = {
            p.obligation.obligation_id
            for p in proposals
            if p.changes_the_register
            and p.decision is Decision.APPROVE
            and p.obligation is not None
        }

        kept = tuple(row for row in register.obligations if row.obligation_id in approved_rows)
        committed = Register(
            obligations=kept,
            conflicts=register.conflicts,
            findings=register.findings,
        )

        rejected = sum(1 for p in proposals if p.decision is Decision.REJECT)
        log = [
            log_entry(
                "commit",
                "committed",
                f"{len(kept)} rows approved, {rejected} item(s) rejected, "
                f"register digest {committed.digest()}",
            )
        ]

        return {
            "register": register_to(committed),
            "committed": True,
            "cost": self._spent(),
            "stage_log": log,
        }
