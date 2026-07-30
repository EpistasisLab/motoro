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

**A run executes end to end.** Create an agent, create a run, execute it under
either of the two migrated execution patterns.

61 files / 11,055 LOC — about 15% of the ARES backend. 26 tests pass against a
real Postgres, including full Sense→Reason→Plan→Act runs under both patterns.

| Landed | |
|---|---|
| SRPA loop | `runtime`, `sense`, `reason`, `plan`, `act`, `phase`, `context` |
| Pattern engine | `base`, `registry`, `orchestrator`, `composition` |
| Patterns | `single_agent_baseline`, `reason_act` — **2 of 37**, added one at a time |
| LLM bridge | `llm_service` + a pluggable credential resolver |
| MCP | `client`, `registry`, `adapters` |
| Persistence | 5 tables, own Alembic chain (`agentic_core.migrations`) |
| Composition root | `agentic_core.runner` — `create_agent`, `create_run`, `execute_run` |
| Observability | tracing + metrics, with a configurable instrument prefix |

Not yet here: the other 35 patterns, semantic/episodic memory, the worker, the
evaluation and scoring subsystems, per-user isolation, users and auth.

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
cp .env.example .env
docker compose up -d                    # postgres :5453, redis :6381

python -m venv .venv && .venv/bin/pip install -e '.[dev]'
set -a && . ./.env && set +a

.venv/bin/pytest
.venv/bin/ruff check src tests scripts examples
.venv/bin/mypy src
.venv/bin/lint-imports                  # core must never import a product
```

### How a product uses core — three phases

`examples/` is laid out as the three phases a product genuinely has. The split
matters: only the third belongs on a request path.

```bash
# ── 1. DEPLOY STEP — once per release, before the app starts ────────────────
python -m agentic_core.migrations upgrade --url "$AGENTIC_DATABASE_URL"

# ── 2. PROVISIONING — once; agents are durable resources ───────────────────
python examples/provision.py            # prints the agent id; idempotent by name

# ── 3. RUNTIME — per request ───────────────────────────────────────────────
python examples/run.py --agent-id <id> --input "What is 17 * 23?"
```

`examples/settings.py` holds the one thing all three share: a `CoreSettings`
subclass with the product's env prefix.

**Migrations are a deploy step, not app startup.** Running them per process means
every replica races the same migration, an API and a worker both try to migrate,
and for a window old and new code run against a half-migrated schema — and a
migration failure becomes a crash-looping app instead of a failed deploy you can
read. So core ships a CLI and *does not* migrate for you. An application should
**verify** the schema and refuse to start if it is behind; both example scripts do
that with `migrations.current_revision()`.

**`create_run` and `execute_run` are separate calls** so a product can enqueue
execution rather than block on it: create the run in the request, return the id,
let a worker execute it. `examples/run.py` does both inline because it is a CLI —
that is the one thing a real product would change.

**The product owns the database connection.** Core never opens a session on its
own: every public entry point takes `db: AsyncSession` as its first argument, and
where that comes from is yours — a FastAPI `Depends(get_db)`, your own
`async_sessionmaker`, whatever your framework already manages. `system_session`
is only core's convenience for contexts with *no* request to scope to (a CLI, a
worker, a cron job); a web app would not use it.

Core does own the *engine* by default — `get_engine()` reads
`CoreSettings.database_url` — so that core and product models, which share one
`Base` and one database, share one connection pool instead of opening two. A
product may ignore it entirely and hand in sessions from its own engine.

> One caveat as more patterns land: 21 of ARES's 37 pattern plugins open their own
> sessions via `models.database`. Neither of the two migrated so far does — they
> use the session the runtime was handed — so "core never opens a session" holds
> today and will need revisiting when the coordination patterns arrive.

**Schema checks are a startup concern, not a per-request one.** A feature
developer never touches `migrations.current_revision()`. It belongs once in the
app's startup path, to refuse to serve against a schema that is behind, and the
examples put it in a clearly-labelled startup section for that reason.

Verified end to end from an empty database: refuse → migrate → provision → run.

### Setting a provider

Put the conventional variables in `.env` and a run works with no code:

```bash
ANTHROPIC_API_KEY=sk-ant-...
# or, for Anthropic on Microsoft Foundry — BOTH are required:
ANTHROPIC_FOUNDRY_API_KEY=...
ANTHROPIC_FOUNDRY_RESOURCE=my-resource
```

**The `AGENTIC_` prefix does not apply to credentials.** They carry bare-name
validation aliases, so `ANTHROPIC_API_KEY` is read as-is — these are conventional
names your tooling already sets, and putting them under a product prefix buys
nothing.

**Foundry needs the resource as well as the key.** The key does not encode an
endpoint; the base URL is derived from the resource name
(`https://<resource>.services.ai.azure.com`, the bare host — litellm's `azure_ai`
route appends `/anthropic/v1/messages` itself, so a base ending in `/anthropic`
double-counts the segment). A key with no resource raises at resolution time
rather than failing later as an opaque 401.

Resolution happens at **call time**, via `services.credentials`, in this order:

1. A credential on the `ModelConfig` (an explicit per-call override).
2. The installed resolver — `env_credential_resolver` (reads settings) unless replaced.
3. Nothing, and the call fails loudly rather than borrowing a shared key.

Why not just set `ModelConfig.api_key`? Because it is `exclude=True`, so it does
**not** survive being persisted with the agent and rebuilt when the run executes —
a key set at agent-creation time is gone by call time. A test pins this.

A product that keeps credentials elsewhere installs its own resolver:

```python
from agentic_core.services.credentials import set_credential_resolver
set_credential_resolver(my_resolver)   # ARES reads an encrypted per-user table
```

### Core's backing services are a requirement

Core requires **PostgreSQL and Redis**. What it does not own is whose servers
they are:

| | Core owns | Product owns |
|---|---|---|
| Postgres | the **schema** — five tables and their migration chain | the server, credentials, deployment |
| Redis | the key layout for per-run working memory | the server, credentials, deployment |

A product provisions the instances and applies core's schema **before** its own
migrations, since product tables routinely carry foreign keys into `agents` and
`agent_runs` and the reverse is forbidden:

```python
from agentic_core.migrations import upgrade
upgrade()                      # core's tables, at head  (or: await upgrade_async())
# ...then the product's own `alembic upgrade head`
```

Core's chain lives **inside the package** (`src/agentic_core/migrations/`) so a
pip-installed core can still be migrated, and stamps
`alembic_version_agentic_core` rather than `alembic_version` — so a product's
chain runs alongside it and neither stamps over the other. Autogenerate is
filtered to core's five tables, because core and a product share one
`Base.metadata` and without the filter core would propose dropping every product
table.

`runner.init_schema()` (`create_all`) remains for tests and scratch work. It is
**not** the production path — nothing is version-tracked. A test asserts the two
paths produce an identical schema, column for column and index for index, so they
cannot silently diverge. To adopt migrations on a database created by
`init_schema`, `migrations.stamp()` it at head.

Redis is required with a *graceful-degradation* path rather than a hard failure:
`AgentRuntime._create_working_memory` pings it and logs `"Redis unavailable —
running without working memory"` if it cannot connect. A run still completes, but
without per-run working memory — a degraded mode, not the intended one.

### `compose.yml` is core's own instances, not a deployment

Core is a headless library — no FastAPI app, no worker entrypoint — so there is
nothing in that file to run it *as*. It exists so core's tests and examples have
their own database and Redis rather than borrowing a product's. **If a `backend:`
or `api:` service ever appears in it, the boundary has slipped.**

- **Services are named `agentic-core-postgres` / `agentic-core-redis`**, not
  `postgres` / `redis`, so a product merging or extending this file finds nothing
  generic to collide with.
- **Ports 5453 / 6381**, offset from the ARES stack (5452/6379/6380) and any
  other local stack, so they can all run at once.
- **Two databases**: `agentic_core` for development, `agentic_core_test` for the
  suite — which drops and recreates its schema per test, so it must not share a
  database with anything you want to keep. Created by
  `docker/initdb/01-test-database.sql` on first boot of an empty volume.
- **`pgvector/pgvector:pg16`**, even though core needs no `Vector` column today.
  Semantic memory will, and the image is a strict superset, so paying now avoids
  recreating the volume later.

`testcontainers` is the better answer for CI (a throwaway database per session,
nothing to remember to start), but it cannot give you a database to inspect
between runs. Worth adding alongside this, not instead of it.
