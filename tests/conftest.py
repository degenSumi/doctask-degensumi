"""What the suite runs against.

The tests prove the system, not whatever is in the developer's `.env`. The
provider is pinned to the pattern-backed stand-in here, before any module that
reads settings at import time is imported, so the suite is keyless and offline
on every machine rather than on the ones that happen to be configured that way.

Environment variables take precedence over `.env`, so this holds even when a
real provider and key are configured for ordinary use.
"""

from __future__ import annotations

import os

os.environ["LLM_PROVIDER"] = "fake"
os.environ.pop("LLM_API_KEY", None)
