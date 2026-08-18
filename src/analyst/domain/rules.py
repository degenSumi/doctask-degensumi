"""Patterns that identify text in a source aimed at the system.

A source document describes the world: what was agreed, what was billed, what
was said. Text that instead addresses the software reading it, redefines its
instructions, or commands its approval workflow is not content about the world.
It is reported as a finding and never acted on.

Detection is deterministic rather than delegated to a model, because a component
that can be reasoned with can be reasoned out of its own defence. False
positives are the direction worth failing in: a wrongly flagged passage costs a
person one glance, a missed one puts hostile text into a prompt.

Patterns are data. A new one is an entry in this table, and its name appears in
the finding so the reason a passage was flagged is always legible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    describes: str


def _compile(source: str) -> re.Pattern[str]:
    return re.compile(source, re.IGNORECASE | re.MULTILINE)


INSTRUCTION_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        name="ignore-previous-instructions",
        regex=_compile(
            r"\b(ignore|disregard|forget)\b[^.]{0,30}?"
            r"\b(previous|prior|earlier|above|preceding|all)\b[^.]{0,30}?"
            r"\b(instruction|prompt|rule|direction|context)s?\b"
        ),
        describes="attempts to discard the instructions the system was given",
    ),
    Pattern(
        name="replacement-instructions",
        regex=_compile(r"\b(new|updated|revised)\s+(instructions?|system\s+prompt)\b\s*[:\-.]"),
        describes="presents itself as a replacement set of instructions",
    ),
    Pattern(
        name="reassigns-role",
        regex=_compile(r"\byou\s+are\s+now\b|\bunrestricted\s+mode\b|\bautonomous\s+mode\b"),
        describes="tells the system it has become something else",
    ),
    Pattern(
        name="addresses-model",
        regex=_compile(r"\byou\s+are\s+(an?\s+)?(ai|assistant|agent|bot|language\s+model|llm)\b"),
        describes="addresses the reader as software rather than describing the matter",
    ),
    Pattern(
        name="names-system-prompt",
        regex=_compile(r"\bsystem\s+prompt\b|\bsystem\s+note\s+for\b"),
        describes="refers to the system's own configuration",
    ),
    Pattern(
        name="suppress-findings",
        regex=_compile(
            r"\b(do\s*n[o']?t|never|no\s+need\s+to)\b[^.]{0,25}"
            r"\b(report|flag|raise|record|mention)\b[^.]{0,25}"
            r"\b(discrepanc|issue|finding|conflict|error)"
        ),
        describes="asks for a discrepancy to go unreported",
    ),
    Pattern(
        name="approve-everything",
        regex=_compile(r"\bapprove\b[^.]{0,25}\b(all|every|everything|each|any)\b"),
        describes="commands the approval workflow rather than describing the matter",
    ),
    Pattern(
        name="bypass-review",
        regex=_compile(
            r"\b(without|skip(ping)?|bypass(ing)?)\b[^.]{0,20}"
            r"\b(review|approval|confirmation|checking)\b"
        ),
        describes="asks for the human review step to be skipped",
    ),
    Pattern(
        name="mass-resolve",
        regex=_compile(r"\bmark\b[^.]{0,25}\b(all|every)\b[^.]{0,25}\b(resolved|done|closed)\b"),
        describes="commands resolution of items other than its own subject",
    ),
    Pattern(
        name="dictates-output",
        regex=_compile(r"\b(reply|respond|answer|output)\b[^.]{0,15}\bonly\s+with\b"),
        describes="dictates what the system must say",
    ),
    Pattern(
        name="chat-control-tokens",
        regex=_compile(r"<\|[^|]{0,40}\|>|\[/?INST\]|<\s*/?\s*(system|assistant)\s*>"),
        describes="contains control tokens from a chat transcript format",
    ),
    Pattern(
        name="forged-turn",
        regex=_compile(r"^\s*(system|assistant)\s*:"),
        describes="imitates a turn in a conversation with the system",
    ),
)


@dataclass(frozen=True, slots=True)
class Detection:
    pattern: Pattern
    start: int
    end: int
    quote: str


def detect_instructions(text: str) -> tuple[Detection, ...]:
    """Every passage in the text that addresses the system.

    All matches are returned rather than the first, because the finding shown to
    a person should account for the whole document, not one sentence of it.
    """
    found: list[Detection] = []
    for pattern in INSTRUCTION_PATTERNS:
        for match in pattern.regex.finditer(text):
            found.append(
                Detection(
                    pattern=pattern,
                    start=match.start(),
                    end=match.end(),
                    quote=match.group(0),
                )
            )
    return tuple(sorted(found, key=lambda d: d.start))


def redact(text: str, detections: tuple[Detection, ...]) -> str:
    """The text with hostile passages replaced by a marker.

    Used when a quarantined source still has to be read for facts. The rest of
    the document may be perfectly ordinary, and discarding it entirely would let
    one hostile sentence hide the content around it.
    """
    if not detections:
        return text

    pieces: list[str] = []
    cursor = 0
    for detection in detections:
        if detection.start < cursor:
            continue
        pieces.append(text[cursor : detection.start])
        pieces.append("[removed: text addressed to the system]")
        cursor = detection.end
    pieces.append(text[cursor:])
    return "".join(pieces)
