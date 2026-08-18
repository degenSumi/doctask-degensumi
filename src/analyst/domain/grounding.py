"""Turns a claimed quote into a verified span, or discards it.

This is where "it never bluffs" is enforced in practice. A model reports a value
and the text it read it from; that text is then located in the source. If it
cannot be found, the answer is dropped rather than recorded, because a value
whose quote is not in the document is an invented one however plausible it reads.

Matching is exact first, then whitespace-insensitive. It deliberately stops
there: fuzzier matching would start accepting quotes the source does not
actually contain, which is the failure this module exists to prevent.
"""

from __future__ import annotations

import re

from analyst.domain.models import Fact, Source, Span

MIN_QUOTE_CHARS = 8
"""Shorter quotes match too much text to identify a location."""


def locate(source: Source, quote: str) -> Span | None:
    """Find `quote` in `source`, returning where it is or nothing.

    Returns None when the quote is absent, too short to identify a place, or
    appears more than once, since an ambiguous citation points nowhere in
    particular.
    """
    cleaned = quote.strip()
    if len(cleaned) < MIN_QUOTE_CHARS:
        return None

    exact = _single_index(source.text, cleaned)
    if exact is not None:
        return Span(
            source_id=source.source_id,
            start=exact,
            end=exact + len(cleaned),
            quote=cleaned,
            locator=locator_for(source.text, exact),
        )

    return _locate_ignoring_whitespace(source, cleaned)


def _single_index(haystack: str, needle: str) -> int | None:
    first = haystack.find(needle)
    if first < 0:
        return None
    if haystack.find(needle, first + 1) >= 0:
        return None
    return first


def _locate_ignoring_whitespace(source: Source, quote: str) -> Span | None:
    """Match a quote whose line breaks or spacing differ from the source.

    Models routinely reflow the text they quote. The words must still be the
    source's words, in the source's order; only the spacing between them is
    allowed to differ.
    """
    words = [re.escape(word) for word in quote.split()]
    if len(words) < 2:
        return None

    pattern = re.compile(r"\s+".join(words))
    matches = list(pattern.finditer(source.text))
    if len(matches) != 1:
        return None

    match = matches[0]
    return Span(
        source_id=source.source_id,
        start=match.start(),
        end=match.end(),
        quote=source.text[match.start() : match.end()],
        locator=locator_for(source.text, match.start()),
    )


_HEADING = re.compile(r"^(#{1,6}\s+.*|\s*\d+\.\s+[A-Z].*)$", re.MULTILINE)


def locator_for(text: str, offset: int) -> str:
    """A human-readable position: the nearest heading above, plus a line number.

    Purely for the reader. Nothing depends on it being right, which is why it is
    allowed to be a heuristic while the span offsets are not.
    """
    line = text.count("\n", 0, offset) + 1

    heading = ""
    for match in _HEADING.finditer(text, 0, offset):
        heading = match.group(0).lstrip("# ").strip()

    return f"{heading} (line {line})" if heading else f"line {line}"


def ground(source: Source, subject: str, value: str, quote: str) -> Fact | None:
    """Build a fact only if its quote is really in the source."""
    span = locate(source, quote)
    if span is None:
        return None
    return Fact(subject=subject, value=value, support=(span,))


def verify_all(facts: tuple[Fact, ...], sources: dict[str, Source]) -> tuple[Fact, ...]:
    """Keep only the facts whose spans still say what they claim.

    Run before a register is published, so a source that changed underneath a
    citation invalidates the claim rather than silently keeping it.
    """
    kept = []
    for fact in facts:
        source_ok = all(
            span.source_id in sources and span.verify(sources[span.source_id])
            for span in fact.support
        )
        if source_ok:
            kept.append(fact)
    return tuple(kept)
