"""agentic-core — a headless agentic platform.

Core provides the Sense-Reason-Plan-Act runtime, the pattern engine, the LLM
bridge, MCP integration, memory, and the persistence beneath them. It provides
**no** ``FastAPI`` application and no routers: a product owns its HTTP layer, its
UI, and its own domain tables, and depends on core for the machinery.

The public surface is deliberately small and grows one slice at a time. Lifecycle
is the only thing exported from the package root:

    from agentic_core import CoreSettings, configure

    class Settings(CoreSettings):
        model_config = SettingsConfigDict(env_prefix="MYAPP_")

    configure(Settings())          # before anything reads a setting

Everything else is imported from its own module (``agentic_core.engine.runtime``,
``agentic_core.services.llm_service``, …). That is intentional: a facade over ~70
services would be delegation code with no other purpose, so the boundary is
enforced by an import-linter contract instead of by indirection.
"""

from agentic_core.config import CoreSettings, configure, get_settings

__all__ = ["CoreSettings", "configure", "get_settings"]

__version__ = "0.0.1"
