"""The seam for language understanding.

Every call is a question about one passage of one document, answered as
structured data. The model is never asked to write the deliverable, and never
sees the register: it reads text and reports what it found, while the code
decides what that means.

The split matters for two reasons. Source documents are written by other people
and arrive untrusted, so the component that reads them holds no authority over
the output. And a claim can only enter the register with a span attached, which
means a model that answers without pointing at text is discarded rather than
believed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from analyst.domain.models import SourceKind


@dataclass(frozen=True, slots=True)
class Extraction:
    """One value the model claims to have found, with the text it read it from.

    `quote` must appear in the passage that was sent. A quote that cannot be
    located is treated as an invented answer and dropped, which is what stops a
    fluent guess reaching the register.
    """

    subject: str
    value: str
    quote: str
    locator: str = ""


@dataclass(frozen=True, slots=True)
class Classification:
    kind: SourceKind
    effective_date: str | None
    quote: str
    """The text the decision was read from, so a wrong classification is legible."""


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    """Whether one rule held, and the text that shows it."""

    rule_id: str
    satisfied: bool
    quote: str
    explanation: str


@dataclass(frozen=True, slots=True)
class Cost:
    """What a call consumed, for the run report."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def plus(self, other: Cost) -> Cost:
        return Cost(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ModelUnavailable(Exception):
    """The model could not be reached or did not answer usably.

    Separate from a model that answered "I don't know", because an outage must
    not read as a finding about the document.
    """


@runtime_checkable
class LanguageModel(Protocol):
    def classify_source(self, filename: str, passage: str) -> Classification:
        """Say what kind of document this is, quoting the text that shows it."""
        ...

    def extract(self, subjects: Mapping[str, str], passage: str) -> tuple[Extraction, ...]:
        """Report the requested values, each quoting the text it came from.

        Subjects are given as name to description. The description exists
        because several subjects are near neighbours in a document — the terms
        an agreement sets and the terms printed on an invoice read almost
        identically — and a name alone leaves the reader to guess which is meant.

        Subjects the passage does not answer are omitted. Returning nothing is a
        valid answer and is preferred to a plausible guess.
        """
        ...

    def check_rule(self, rule_id: str, statement: str, passage: str) -> RuleVerdict:
        """Say whether the passage satisfies the rule, quoting the relevant text."""
        ...

    @property
    def cost(self) -> Cost:
        """What this model has consumed so far."""
        ...
