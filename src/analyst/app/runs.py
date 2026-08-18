"""Naming a run after what it reads.

A run is identified by its inputs rather than by a fresh random id, so pointing
any surface at the same corpus twice continues the work instead of paying to
read the same documents again. Starting over stays available, but it is asked
for rather than being what happens by default.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


def run_id_for(corpus: Path, rules: Path, *, fresh: bool = False) -> str:
    """The run that reading this corpus under these rules belongs to.

    Paths are resolved, so the same folder named two ways is one run. The rules
    file is part of the key because the same documents checked against different
    rules is a different question and deserves its own answer.
    """
    key = f"{corpus.resolve()}|{rules.resolve()}".encode()
    identifier = f"run-{hashlib.blake2b(key, digest_size=4).hexdigest()}"
    return f"{identifier}-{uuid.uuid4().hex[:6]}" if fresh else identifier
