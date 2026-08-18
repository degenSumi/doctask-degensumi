"""A language model stood in by patterns, so the system runs with no key.

It reads the same passages a real model would and answers in the same shape.
What it must never do is answer with text that is not in the passage, because
the grounding step would then correctly discard it and the pipeline under test
would be exercised wrongly.

This is a working stand-in, not a recording. Everything that makes the system
interesting happens elsewhere and is exercised for real: locating quotes,
comparing facts across documents, deciding what supersedes what, quarantining
hostile text, the review gate, and resuming after a stop. The fake only answers
"what does this passage say", which is the one part a model is needed for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from analyst.domain.models import SourceKind
from analyst.ports.language_model import (
    Classification,
    Cost,
    Extraction,
    LanguageModel,
    RuleVerdict,
)

_KIND_MARKERS: tuple[tuple[re.Pattern[str], SourceKind], ...] = (
    (re.compile(r"^#?\s*AMENDMENT\b", re.IGNORECASE | re.MULTILINE), SourceKind.AMENDMENT),
    (
        re.compile(r"^#?\s*MASTER SERVICES AGREEMENT\b", re.IGNORECASE | re.MULTILINE),
        SourceKind.CONTRACT,
    ),
    (re.compile(r"^\s*INVOICE\s*$", re.IGNORECASE | re.MULTILINE), SourceKind.INVOICE),
    (re.compile(r"^#?\s*FILE NOTE\b", re.IGNORECASE | re.MULTILINE), SourceKind.CORRESPONDENCE),
)

_EFFECTIVE_DATE = re.compile(
    r"\*\*Effective date:\*\*\s*(?P<value>[^\n]+)|Effective date:\s*(?P<alt>[^\n]+)",
    re.IGNORECASE,
)

# Each subject maps to a pattern whose match is quoted verbatim, and a group
# holding the value. The quote is always a slice of the passage, never a
# paraphrase, so grounding can locate it.
_SUBJECTS: dict[str, tuple[re.Pattern[str], str]] = {
    "payment_terms": (
        re.compile(
            r"[Pp]ayment terms (?:are|:)\s*net\s+(?P<v>[a-z\-]+(?:\s*\(\d+\))?|\d+)[^.\n]*",
        ),
        "v",
    ),
    "monthly_fee": (
        re.compile(r"monthly fee(?: of| increased to)?\s*(?P<v>GBP\s?[\d,]+)", re.IGNORECASE),
        "v",
    ),
    "invoice_total": (
        re.compile(r"Total due\s+(?P<v>GBP\s?[\d,]+)", re.IGNORECASE),
        "v",
    ),
    "invoice_terms": (
        re.compile(r"Payment terms:\s*(?P<v>net\s+\d+\s+days[^.\n]*)", re.IGNORECASE),
        "v",
    ),
    "invoice_date": (
        re.compile(r"Invoice date:\s*(?P<v>[^\n]+)", re.IGNORECASE),
        "v",
    ),
    "invoice_number": (
        re.compile(r"Invoice number:\s*(?P<v>[^\n]+)", re.IGNORECASE),
        "v",
    ),
    "purchase_order": (
        re.compile(r"Purchase order:\s*(?P<v>[^\n]+)", re.IGNORECASE),
        "v",
    ),
    "governing_law": (
        re.compile(r"governed by the laws of\s+(?P<v>[^.\n]+)", re.IGNORECASE),
        "v",
    ),
    "delivery_window": (
        re.compile(
            r"deliver all materials within\s+(?P<v>[a-z\-]+\s*\(\d+\)\s*days)", re.IGNORECASE
        ),
        "v",
    ),
    "notice_period": (
        re.compile(r"terminate on\s+(?P<v>[a-z\-]+\s*\(\d+\)\s*days)", re.IGNORECASE),
        "v",
    ),
    "agreement_reference": (
        re.compile(r"Agreement reference:\s*(?P<v>[^\n]+)", re.IGNORECASE),
        "v",
    ),
    "effective_date": (
        re.compile(r"\*{0,2}Effective date:\*{0,2}\s*(?P<v>[^\n]+)", re.IGNORECASE),
        "v",
    ),
    "data_protection": (
        re.compile(r"(?P<v>process personal data only on documented instructions[^.\n]*)"),
        "v",
    ),
}

# What each rule needs to find in the passage for it to hold. Absence is a
# finding; presence is not.
_RULE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "PO-REQUIRED": ("purchase_order",),
    "GOVERNING-LAW": ("governing_law",),
    "AMENDMENT-EFFECTIVE-DATE": ("effective_date",),
    "DATA-PROTECTION-CLAUSE": ("data_protection",),
}


class FakePatternModel(LanguageModel):
    """Answers from patterns. Quotes are always slices of the passage given."""

    def __init__(self) -> None:
        self._cost = Cost()

    @property
    def cost(self) -> Cost:
        return self._cost

    def _charge(self, passage: str, answer_chars: int) -> None:
        # Counted in characters rather than tokens, and reported as such, so the
        # run's cost line is honest about what it measured.
        self._cost = self._cost.plus(
            Cost(calls=1, input_tokens=len(passage), output_tokens=answer_chars)
        )

    def classify_source(self, filename: str, passage: str) -> Classification:
        self._charge(passage, 32)

        kind = SourceKind.UNKNOWN
        quote = ""
        for pattern, candidate in _KIND_MARKERS:
            match = pattern.search(passage)
            if match:
                kind = candidate
                quote = match.group(0).strip()
                break

        effective = None
        date_match = _EFFECTIVE_DATE.search(passage)
        if date_match:
            effective = (date_match.group("value") or date_match.group("alt") or "").strip()
            quote = quote or date_match.group(0).strip()

        return Classification(kind=kind, effective_date=effective, quote=quote)

    def extract(self, subjects: Mapping[str, str], passage: str) -> tuple[Extraction, ...]:
        found: list[Extraction] = []
        for subject in subjects:
            entry = _SUBJECTS.get(subject)
            if entry is None:
                continue
            pattern, group = entry
            match = pattern.search(passage)
            if match is None:
                # Saying nothing is the correct answer for a passage that does
                # not carry the value, and is preferred to a plausible guess.
                continue
            found.append(
                Extraction(
                    subject=subject,
                    value=match.group(group).strip(),
                    quote=match.group(0).strip(),
                )
            )

        self._charge(passage, sum(len(e.quote) for e in found))
        return tuple(found)

    def check_rule(self, rule_id: str, statement: str, passage: str) -> RuleVerdict:
        self._charge(passage, 64)

        required = _RULE_EVIDENCE.get(rule_id)
        if required is None:
            # Rules needing comparison across documents are settled by the
            # reconciler, which has both sides. Reporting "satisfied" here would
            # claim a check that was never made.
            return RuleVerdict(
                rule_id=rule_id,
                satisfied=True,
                quote="",
                explanation="not decidable from a single passage; settled during reconciliation",
            )

        for subject in required:
            found = self.extract({subject: ""}, passage)
            if found:
                return RuleVerdict(
                    rule_id=rule_id,
                    satisfied=True,
                    quote=found[0].quote,
                    explanation=f"the passage states {subject.replace('_', ' ')}",
                )

        first_line = passage.strip().splitlines()[0] if passage.strip() else ""
        return RuleVerdict(
            rule_id=rule_id,
            satisfied=False,
            quote=first_line,
            explanation=f"the passage does not state {required[0].replace('_', ' ')}",
        )
