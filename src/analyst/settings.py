"""Configuration read from the environment.

Credentials are never accepted as arguments, written to disk, or logged. The
model provider is chosen here and nowhere else, so every entry point — the CLI,
the API, the MCP server — runs against the same one.

The default provider is the pattern-backed stand-in, which means a fresh clone
runs completely with no key at all. Supplying a key changes which adapter is
built and nothing else.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from analyst.ports.language_model import LanguageModel


class Provider(StrEnum):
    FAKE = "fake"
    """Patterns, no network. What the tests and a fresh clone use."""

    GEMINI = "gemini"
    GROQ = "groq"


class MissingCredential(Exception):
    """Raised when a provider was chosen without the key it needs."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Provider = Provider.FAKE
    llm_api_key: SecretStr = Field(default=SecretStr(""))
    """Wrapped so printing the settings object cannot leak it."""

    llm_model: str = ""

    checkpoint_path: Path = Path("runs.db")
    corpus_dir: Path = Path("corpus")
    inbox_dir: Path = Path("inbox")
    rules_path: Path = Path("corpus/rules/vendor-billing-checklist.yaml")

    @property
    def has_key(self) -> bool:
        return bool(self.llm_api_key.get_secret_value().strip())

    def build_model(self) -> LanguageModel:
        """The model this configuration asks for.

        A provider chosen without a key is refused here rather than at the first
        call, so the failure names the fix instead of surfacing mid-run.
        """
        if self.llm_provider is Provider.FAKE:
            from analyst.adapters.fake_model import FakePatternModel

            return FakePatternModel()

        if not self.has_key:
            raise MissingCredential(
                f"LLM_PROVIDER is {self.llm_provider} but LLM_API_KEY is not set. "
                "Copy .env.example to .env and add a key, or leave LLM_PROVIDER unset "
                "to run on the built-in pattern model."
            )

        from analyst.adapters.hosted_model import HostedModel

        return HostedModel(
            provider=self.llm_provider,
            api_key=self.llm_api_key.get_secret_value(),
            model=self.llm_model,
        )
