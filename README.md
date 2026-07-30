# agentic-core

A **headless agentic platform**: the Sense-Reason-Plan-Act runtime, the pattern
engine, the LLM bridge, MCP integration, memory, and the persistence beneath
them.

Core ships **no `FastAPI` application, no routers, and no UI**. A product owns its
HTTP layer, its frontend, and its own domain tables, and depends on core for the
machinery. Two products are planned on top of it: `ares` (an experiment harness
whose subject is the agent pipeline) and `ecoxai`.

Extracted from [ARES](https://github.com/EpistasisLab/ARES) one slice at a time.
The scope decisions and the boundary contract live in ARES under
`project_plan/CORE_SPLIT_INVENTORY.md` and `project_plan/AGENTIC_CORE_BOUNDARY.md`.

## Status

Slice 1 — the SRPA loop — in progress. **Step 0 of 6 landed.**

| Step | Contents | LOC | State |
|---|---|---|---|
| 0 | `config`, `models.base`, `observability.*`, `schemas.{llm,pattern}`, `security.prompt_injection`, `services.{credential_scrubber,llm_errors,model_capabilities,retry}` | 1,914 | ✅ |
| 1 | `mcp.client`, `models.{agent,database,pricing,redis,run}`, `schemas.{agent,pricing}` | 1,390 | — |
| 2 | `engine.context`, `mcp.registry`, `services.pricing_service` | 783 | — |
| 3 | `engine.phase`, `mcp.adapters` | 612 | — |
| 4 | `engine.runtime`, `engine.sense`, `services.llm_service` | 2,059 | — |
| 5 | `engine.act`, `engine.plan`, `engine.reason` | 948 | — |

Slice 1 totals **31 modules / 7,963 LOC** — 11% of the ARES backend. It has no
dependency cycles, so the steps above are a strict order rather than a
suggestion.

### Core does not manage users

Core owns agents, runs, steps, tools, memory, and patterns — the Agent Runtime
and Service Layer of the ARES architecture diagram. It has **no** user model, no
authentication, no tokens, and no ownership columns. That matches the data model
documented in ARES `docs/ARCHITECTURE.md`, whose core entities carry
`created_at`/`updated_at` and no owner at all; the ownership columns arrived
later, with per-user isolation.

Consequences, each of which is a decision rather than an omission:

- **No `OwnedMixin`.** The ARES original adds `created_by_id` / `updated_by_id`
  as `ForeignKey("users.id")`. Since `users` would be a *product* table, that is
  a core→product foreign key, which the boundary forbids.
- **No `UserSummary`.** It is an attribution DTO for API responses, and core has
  no API to respond with.
- **`agents.created_by_id` and `agent_runs.started_by_id` do not come across.**
  Both are `NOT NULL` in ARES, so importing them would mean core could not store
  an agent or a run without a user.

A product that wants ownership adds it on its own side. The recommended shape is
an opaque, nullable, un-constrained `owner_id` on the core model, with the
product's migration supplying the foreign key, the `NOT NULL`, and any per-owner
unique constraint. The column is declared in core rather than added purely by a
product migration for a mechanical reason: core's Alembic autogenerate diffs
against `Base.metadata`, so a column in the database but absent from core's model
is one core proposes to **drop**.

### Deliberately excluded from slice 1

Each of these was measured as separable, not assumed to be:

- **Per-user isolation** (`security/isolation/*`, `engine/scoping.py`) — whether
  core needs it at all is undecided. `engine/scoping.py` is imported only by
  pattern plugins, never by the SRPA loop, so it can wait for them.
- **Per-user credential resolution** — `llm_service._resolve_connection_for_user`
  is the sole coupling to `user_llm_settings`, it short-circuits when
  `ModelConfig.api_key` is set, and all its imports are function-local. It
  becomes a registered resolver hook; core's default reads the model config.
- **Agent-relationship resource limits** — `runtime.py` already wraps
  `check_resource_limits` in a `try` that returns `None` on failure, so it is
  optional by design. Becomes a hook.
- **The 37 pattern plugins** — `engine/patterns/builtin/__init__.py` is one line;
  plugins are lazily imported by `discover()`, so they are individually
  deferrable. That defers ~16,200 LOC and lets each pattern be added and tested
  on its own.

## Layout

```
src/agentic_core/
  config.py           CoreSettings + configure() + the settings proxy
  models/             SQLAlchemy models; models/base.py owns the shared Base
  schemas/            Pydantic contracts
  services/           LLM bridge, cost, retry, scrubbing
  mcp/                MCP client, registry, tool adapters
  engine/             the SRPA loop, and later the pattern engine
  observability/      tracing + metrics
  security/           prompt-injection fencing (isolation is undecided)
scripts/
  pull_from_ares.py   the migration tool — see below
tests/
```

## Consuming core from a product

Core's only package-root export is lifecycle. Everything else is imported from
its own module; there is no facade, because a facade over ~70 services would be
delegation code with no other purpose. The boundary is enforced by an
import-linter contract instead.

```python
from pydantic_settings import SettingsConfigDict
from agentic_core import CoreSettings, configure

class Settings(CoreSettings):
    model_config = SettingsConfigDict(env_prefix="ARES_", env_file=".env", extra="ignore")
    # product-only fields here

configure(Settings())   # before any core module reads a setting
```

`configure()` raises if called after settings have already been read — half a
process on defaults and half on the product's values is much harder to diagnose
than a startup failure.

Nothing in core hardcodes a product's identity. `otel_service_name` and
`metrics_prefix` both default to `agentic-core`, so a product that forgets to
override them will at least not claim to be a different product.

## Migration tool

```bash
export ARES_SRC=~/dev/ARES/backend/src/ares

python scripts/pull_from_ares.py --step 1          # pull a dependency-ordered step
python scripts/pull_from_ares.py models.agent      # or named modules
python scripts/pull_from_ares.py --verify          # check what is on disk
```

`REQUIRED_EDITS` in that script lists every module that must **not** be copied
verbatim, with what has to change — the ownership columns on `models.agent` and
`models.run`, `scoped_session` on `models.database`, the credential resolver in
`services.llm_service`. Pulling one prints the note. That list exists because
`models/base.py` was pulled mechanically and brought `OwnedMixin` — two
`ForeignKey("users.id")` columns — into a core that does not manage users. A
rewritten import is visible in a diff; a schema dependency smuggled in by a mixin
is not.

`--verify` asserts two invariants over everything already pulled, and should pass
before the next step starts:

1. **No module references `ares` in code.** Docstrings are exempt (prose about
   the split legitimately names ARES); string defaults, identifiers, and
   instrument names are not. `# ares-ok` opts a line out.
2. **Every internal import resolves** to a module that has actually been pulled,
   so each step is self-contained.

Rewriting imports is the easy half. The check exists for the other half — a
hardcoded `"ares-backend"` default or an `ares_` metric name survives a
mechanical pass and quietly makes core impersonate a product.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/ruff check src tests
.venv/bin/mypy src
.venv/bin/lint-imports          # core must never import a product
```
