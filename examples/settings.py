"""The product's settings class — the one thing every entry point shares.

A product subclasses :class:`CoreSettings`, picks its own environment prefix, and
adds whatever fields it needs. Core deliberately sets no prefix of its own: that
is a product concern, and core must not presume ``ARES_`` or anything else.

Provider credentials are the exception. They carry bare-name validation aliases
in ``CoreSettings``, so ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_FOUNDRY_API_KEY`` /
``ANTHROPIC_FOUNDRY_RESOURCE`` are read as written, *without* the prefix — those
are conventional names your tooling already sets.
"""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from agentic_core import CoreSettings


class Settings(CoreSettings):
    """Stands in for a real product's settings class."""

    model_config = SettingsConfigDict(env_prefix="AGENTIC_", env_file=".env", extra="ignore")

    # A real product adds its own fields here — its API host, its feature flags,
    # its own domain configuration. Core never sees them.
