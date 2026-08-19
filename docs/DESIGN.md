# Design

The architectural boundary between core and a product, and the reasoning
behind each piece of it. See the [README](../README.md) for installing and
running core; this doc is the "why."

## Core does not manage users

Core owns agents, runs, steps, tools, memory, and patterns. It has **no** user
model, no authentication, no tokens, and no ownership columns.

Consequences, each of which is a decision rather than an omission:

- **No `OwnedMixin`.** A mixin adding `created_by_id` / `updated_by_id` as
  `ForeignKey("users.id")` would be a core→product foreign key, since `users`
  is a product table — the boundary forbids that.
- **No `UserSummary`.** It is an attribution DTO for API responses, and core has
  no API to respond with.
- **No `created_by_id`/`started_by_id` on agents or runs.** A `NOT NULL` owner
  column would mean core could not store an agent or a run without a user,
  which core has no model for.

A product that wants ownership keeps its `users` table in its own database and
passes an **opaque, nullable, un-constrained `owner_id`** to `create_agent` /
`create_run`. Core stores and filters on it (`list_agents(owner_id=...)`) and
never resolves it — it is a `UUID`, not a foreign key, and cannot become one:
`users` lives in a different database. Enforcing that the owner exists is the
product's job, on its own side of the boundary.

## Deliberately excluded, for now

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

## Migrations are a deploy step, not app startup

Running them per process means every replica races the same migration, an API
and a worker both try to migrate, and for a window old and new code run
against a half-migrated schema — and a migration failure becomes a
crash-looping app instead of a failed deploy you can read. If a product wants
to refuse to serve against a schema that is behind,
`migrations.current_revision()` belongs once in its startup path — not in a
request, and not in the examples, because a feature developer should never
have to think about core's schema.

`motoro-migrate` is that deploy step, not an exception to it: a
**one-shot init container** that waits for Postgres to be healthy, runs
`deploy`, and exits 0. One container, running to completion, gating what
follows — there is nothing to race. `docker compose up -d` therefore brings the
database up *ready*, with no second command to remember. It shows as `Exited
(0)` in `docker compose ps`, which is success. A product using its own
orchestrator reproduces the shape with an init container, a Helm hook, or a
release task.

`deploy` is two idempotent steps: `upgrade` applies the schema, then
`sync-catalog` projects the pattern registry into `architectural_patterns`
(see "Pattern metadata" below). Either can be run alone.

The image is ~624 MB because of that second step: building the catalog means
importing every pattern plugin to read its metadata, and the plugins import
the runtime — litellm, instructor, opentelemetry. Applying the *schema* alone
needs none of that, and a boundary test keeps it that way, so splitting this
back into a slim schema-only container remains a one-line change rather than
an investigation.

**Every revision is re-runnable.** Four properties, each with a test:

| Property | Why it can fail |
|---|---|
| `upgrade` twice is a no-op | Trivially true via the version table — pinned anyway, since a product may call it on every boot |
| `downgrade` → `upgrade` reproduces the schema exactly | The real hazard. Autogenerate emits `CREATE TYPE` for a native enum but never the matching `DROP TYPE`, so the baseline could be migrated once and never again (`type "pattern_category" already exists`) |
| `downgrade` leaves no orphan enum types | Dropping a table does not cascade to its enum — types are schema-level objects |
| No revision uses `CONCURRENTLY` | Postgres has transactional DDL and Alembic wraps each revision, so a failed revision rolls back whole, version stamp included. `CONCURRENTLY` opts out of that and can leave a revision half-applied |

The last two are static checks over every revision file, so a future migration
cannot quietly reintroduce either. Both were mutation-tested — they fail when
the enum drops are removed, and when a single enum is dropped from the list.

**`create_run` and `execute_run` are separate calls** so a product can enqueue
execution rather than block on it: create the run in the request, return the
id, let a worker execute it. `examples/run.py` does both inline because it is
a CLI — that is the one thing a real product would change.

## Two databases: core owns its own, entirely

**Core manages its own database. A product never touches it.** No session
appears anywhere in core's public API — no `AsyncSession` parameter, no
`system_session`, no `Depends(get_db)`. A product configures `database_url`
once through settings and then only calls functions. Core opens, commits, and
closes its own connections.

That means **two databases**, not one:

| | Core's database | The product's database |
|---|---|---|
| Contains | agents, runs, steps, and core's own tables | the product's domain tables |
| Schema applied by | `python -m motoro.migrations upgrade` | the product's own Alembic chain |
| Reached through | core's function API only | the product's own sessions |

So that products never need to query core's tables to read their own data
back, the runner exposes reads as well as writes: `get_agent`,
`get_agent_by_name`, `list_agents`, `get_run`, `get_runs`, `list_runs`,
`get_run_steps`. Returned ORM objects are usable after core closes its
session — core's sessionmaker sets `expire_on_commit=False` — so a product can
read `run.status` and `run.token_usage` without a live connection.

**This costs three things, and they are real:**

1. **No cross-database foreign keys.** A product row that refers to a run
   stores an opaque `UUID` with no `ForeignKey`, so the database will not
   enforce that the run exists or cascade its deletion. In practice this is
   felt in few places: most tables that would want a foreign key into
   `agents` / `agent_runs` / `run_steps` are core's own; a product typically
   only needs one from a handful of its own tables (a plan, an experiment
   record, and similar).
2. **No cross-database joins.** Instead of joining product rows against
   `agent_runs`, a product collects run ids and calls `get_runs(ids)` — one
   extra round trip, and an N+1 if written carelessly. That is why `get_runs`
   takes a collection rather than making callers loop.
3. **No cross-database transactions.** Writing a product row and a core run is
   two commits, not one, so a crash between them leaves the product row
   pointing at nothing (or nothing pointing at a run). Products that care need
   a reconciliation path — an unreferenced run is the safer of the two
   orderings.

The trade this buys: core can change its schema, its session strategy, its
connection pooling, and its transaction boundaries without a product noticing,
and no product can corrupt core's tables by writing them directly.

Verified end to end from an empty database: migrate → provision → run.

Core's chain lives **inside the package** (`src/motoro/migrations/`) so
a pip-installed core can still be migrated, and stamps
`alembic_version_motoro` rather than `alembic_version`. Both of those
still matter even with separate databases: the version table name keeps a
*co-located* deployment — the two chains pointed at one database, which core
does not require but does not forbid — from stamping over each other, and
autogenerate stays filtered to core's tables because core and a product share
one `Base.metadata` in the same Python process regardless of where the rows
live.

`runner.init_schema()` (`create_all`) remains for tests and scratch work. It
is **not** the production path — nothing is version-tracked. A test asserts
the two paths produce an identical schema, column for column and index for
index, so they cannot silently diverge. To adopt migrations on a database
created by `init_schema`, `migrations.stamp()` it at head.

Redis is required with a *graceful-degradation* path rather than a hard
failure: `AgentRuntime._create_working_memory` pings it and logs `"Redis
unavailable — running without working memory"` if it cannot connect. A run
still completes, but without per-run working memory — a degraded mode, not
the intended one.

## Pattern metadata: the plugin class is the source of truth

`create_agent` validates `pattern_config` and raises `PatternConfigError` on a
bad slug, bad params, an unsatisfiable dependency, a multi-agent pattern with
no role, or params naming an inactive pattern. A typo fails at creation rather
than after a run exists and a model has been billed. Each pattern's
`configuration_schema` defaults are also merged into `pattern_params` before
`configure()` runs, so creating an agent with no parameters still gets the
documented ones.

**Both read the registry, not the database.** That is the whole design:

```python
class ReasonActPlugin(PatternPlugin):
    slug = "reason_act"
    category = PatternCategory.EXECUTION
    display_name = "ReAct"
    description = "Alternates between deliberation and execution..."
    complexity_phase = PatternPhase.INTROSPECTIVE
    dependencies = ["single_agent_baseline"]
    configuration_schema = {...}          # every property carries a `default`
```

A common alternative is keeping this in a database table, seeded by a data
migration — that costs real bugs:

- **Validation depends on a migration having run.** An unseeded table rejects
  *every* agent with "Unknown pattern slug". Core has a test that creates an
  agent successfully against a deliberately emptied table.
- **Code and catalog drift.** Renaming a pattern's slug leaves any stored
  dependency array pointing at the old name unless a migration chases it
  through data by hand. A test now fails if any declared dependency names an
  unregistered pattern.
- **Defaults exist twice.** The schema the UI renders and the
  `params.get(key, literal)` fallback inside `configure()` are two copies free
  to disagree. A test asserts they match.
- **`is_implemented` is hand-maintained**, with nothing reconciling it against
  which plugins exist. Here it is derived: the class is discoverable, so the
  pattern is implemented.

Declaring `dependencies` on the plugin class also means composition is checked
at agent-creation time: a pattern that depends on another in its own
singleton category (`reason_act` on `single_agent_baseline`) fails fast with
"requires X, which is not active" instead of surfacing as unreachable
auto-resolution code somewhere the model has already been billed.

**Core never reads the table.** `sync_pattern_catalog()` writes it, one way,
as a deploy step, for products that need to query the catalog — a patterns
page, an advisor's knowledge base, an experiment designer choosing factors.
Read it back with `list_catalog()` so a product needs no session of its own:

```python
from motoro.services.pattern_catalog import list_catalog
rows = await list_catalog(implemented_only=True)
```

Rows for patterns that are not registered are reported as `stale` and left
alone rather than deleted: core ships 2 of 37 patterns, so "not registered in
this process" does not mean "does not exist". The upsert is `DO UPDATE`, not
`DO NOTHING` — a description or schema changed in code corrects the row
automatically on the next `sync_pattern_catalog()`, rather than needing a
hand-written migration to chase it through data.

## Memory: episodic and semantic, no owner

Working memory (per-run, Redis) has been here from the start. Episodic
(per-agent run summaries) and semantic (global/per-agent knowledge) are backed
by one `memory_entries` table with a pgvector `embedding` column, behind
`MemoryService`:

```python
from motoro.services.memory_service import MemoryService
from motoro.services.llm_service import LLMService

memory_service = MemoryService(llm_service=LLMService())
result = await execute_run(run_id=run.id, memory_service=memory_service)
```

Both are **opt-in per agent**, off by default:
`create_agent(..., memory_config={"episodic_memory_enabled": True})`.

**No owner column.** A memory's identity is already `agent_id` (which agent it
belongs to) and, for episodic entries, `run_id` (which run produced it) — both
real foreign keys, since `agents`/`agent_runs` are core's own tables in the
same database. Nothing else is needed to know what a memory is. A per-user
isolation feature (scoping memory rows to whoever created them) is out of
scope for core and would need an owner column core otherwise has no use for —
a product enforcing that filters on the *agent's* `owner_id` instead, the same
way it would for any other agent-scoped data.

An owner column also invites a specific failure mode this design avoids:
skipping storage entirely when a run has no acting user, because nothing
satisfies a `NOT NULL` constraint. With no owner concept, storage is
unconditional — an anonymous or system-triggered run keeps its memory too.

**Embedding credentials resolve from settings, not a per-user table.** Core
has no users table to join against, so `EmbeddingService` reads
`CoreSettings.openai_api_key` directly — the same shape as the LLM bridge's
credential resolver, for the same reason. An explicit `api_key` argument
still takes precedence, for a product with its own credential store.

**Default backend is local, no API key needed:**
`embedding_model = "sentence-transformers/BAAI/bge-base-en-v1.5"` runs
in-process via `sentence-transformers`. Any other model name is a remote
litellm-supported embedding call instead. The local default is why
`sentence-transformers` (and its `torch` dependency) installs eagerly for
every product — worth knowing if disk footprint matters and a product only
ever uses remote embeddings.

A bug worth naming, because it explains what "the runtime actually uses it"
means as a test: `PatternOrchestrator.run` builds its own `RunContext` rather
than reusing `AgentRuntime.run`'s, and that mirror was missing
`context.memory_config_data = ...`. Since `execute_run` always wraps the loop
in a `PatternOrchestrator` — for both shipped patterns — Sense read
`episodic_memory_enabled` as `False` on every real run regardless of the
agent's config. Storage still worked, because
`AgentRuntime._episodic_memory_enabled` reads the runtime's own config, not
the context — so memory was written on every run and recalled on none of
them. Fixed here; `test_a_second_run_actually_recalls_the_first_runs_memory`
is the regression test, and it was verified to fail against the reverted bug
before being kept.

## MCP: the client was already done; this adds persistence

`motoro.mcp` (`client`, `registry`, `adapters`) is transport — connect,
discover tools, call a tool — wired into the Act phase from the start. What
was missing is the other half: remembering *which* servers a product uses, so
a fresh process — a worker, a restarted API, a new script invocation —
doesn't have to re-register them by hand. That's `services.mcp_service` +
`models.mcp_server.MCPServerConfig`.

```python
from motoro.services.mcp_service import register_server, hydrate_registry

# Once: connect and persist.
config = await register_server(name="search", transport="stdio", command="npx -y some-search-server")

# Every subsequent process: reconnect from the table, not from code.
await hydrate_registry()
```

**The database is authoritative here — the opposite direction from
`engine.patterns.catalog`.** There, plugin code was the source of truth and
the table was a read-only projection for products to query. Here, the
in-memory `MCPServerRegistry` is the derived, disposable thing: it starts
empty every process and gets rebuilt from `mcp_server_configs`, never the
other way around. `hydrate_registry()` is the function that makes persisting
a server worth doing at all — without it, a registered config would just sit
in the table, never read back into a live connection.

**`owner_id`, same severance as `Agent`/`AgentRun`/`MemoryEntry`** —
`created_by_id: NOT NULL FK -> users` becomes opaque, nullable, no foreign
key. One category of field is dropped outright rather than made opaque:
anything recording which product *feature* proposed or created a resource
(e.g. a `source_plan_id` naming which planning tool suggested this server) —
that's a fact about a product feature, not about the server, and an opaque
UUID would still encode a product concept core has no business referencing at
all.

Two self-contained security modules have zero coupling to anything user- or
product-specific: `security.mcp_command_allowlist` (a stdio command's
executable must be one of `python`/`node`/`npx`/…, and no shell
metacharacters) and `security.ssrf_guard` (an http/sse URL must not resolve
to a private/reserved IP, guarding against DNS rebinding too). Core has no API
schema layer, so `register_server`/`update_server` call it directly — the one
place every registration passes through regardless of caller.

`services.encryption` (Fernet, for a registered server's auth headers) is
**not** a per-user secret like the embedding-credential lookup it sits next
to in spirit — one server-side key (`CoreSettings.encryption_key`), the same
for every row core encrypts. That's what makes it portable with no
adaptation.

### A real bug, found by finally testing this against a live server

No existing test had ever connected `MCPClient` to an actual MCP server
before these tests did. Two defects surfaced immediately, both now fixed and
regression-tested:

- **`mcp>=1.27.1` (an unbounded floor) resolves to `mcp==2.0.0` on a fresh
  install, and 2.0.0 hangs `MCPClient.connect()` forever.** That method always
  installs a `message_handler` on `ClientSession` (to catch
  `notifications/tools/list_changed`), and 2.0.0 changed the background
  message-reading task's lifecycle so the process never exits. Verified
  directly: an identical connect-then-list-tools call returns immediately
  under `1.27.1` and hangs indefinitely under `2.0.0`. `pyproject.toml` now
  caps `mcp>=1.27.1,<2.0.0`.
- **`MCPServerRegistry.disconnect_all()`'s `asyncio.gather()` broke anyio's
  cancel-scope contract.** Each stdio client's transport is an anyio task
  group, and anyio requires cancel scopes to be exited in exact LIFO order, by
  the same task that entered them. `gather()` violated both: it schedules
  each `disconnect()` as its own task, and gave no ordering guarantee at all.
  Fixed to disconnect sequentially, in reverse registration order — verified
  directly against a live pair of stdio servers, where the old
  forward/concurrent version reliably raises `RuntimeError: Attempted to exit
  cancel scope in a different task than it was entered in` and the new
  version does not. The cost is real: shutdown time is now the sum of each
  disconnect, not the max.

Both were invisible until a live server was actually exercised — worth
knowing if you're deciding whether to trust an unbounded dependency range or
a concurrency optimization you haven't load-tested.

## Resolution order for credentials

Resolution happens at **call time**, via `services.credentials`, in this
order:

1. A credential on the `ModelConfig` (an explicit per-call override).
2. The installed resolver — `env_credential_resolver` (reads settings) unless
   replaced.
3. Nothing, and the call fails loudly rather than borrowing a shared key.

Why not just set `ModelConfig.api_key`? Because it is `exclude=True`, so it
does **not** survive being persisted with the agent and rebuilt when the run
executes — a key set at agent-creation time is gone by call time. A test pins
this.

**Foundry needs the resource as well as the key.** The key does not encode an
endpoint; the base URL is derived from the resource name
(`https://<resource>.services.ai.azure.com`, the bare host — litellm's
`azure_ai` route appends `/anthropic/v1/messages` itself, so a base ending in
`/anthropic` double-counts the segment). A key with no resource raises at
resolution time rather than failing later as an opaque 401.

## `compose.yml` is core's own instances, not a deployment

Core is a headless library — no FastAPI app, no worker entrypoint — so there
is nothing in that file to run it *as*. It exists so core's tests and
examples have their own database and Redis rather than borrowing a product's.
**If a `backend:` or `api:` service ever appears in it, the boundary has
slipped.**

- **Services are named `motoro-postgres` / `motoro-redis` /
  `motoro-migrate`**, not `postgres` / `redis`, so a product merging or
  extending this file finds nothing generic to collide with.
- **`motoro-migrate` is the one non-service service**: it applies core's
  schema and exits. Core owns the schema, so applying it is core's job — that
  is not the same as core shipping an app. The invariant above still holds.
- **Ports 5453 / 6381**, offset from Postgres/Redis's own defaults (5432/6379)
  so this can run alongside another local stack without colliding.
- **Two databases**: `motoro` for development, `motoro_test` for
  the suite — which drops and recreates its schema per test, so it must not
  share a database with anything you want to keep. Created by
  `docker/initdb/01-test-database.sql` on first boot of an empty volume.
- **`pgvector/pgvector:pg16`**, even though core needs no `Vector` column
  today. Semantic memory will, and the image is a strict superset, so paying
  now avoids recreating the volume later.

`testcontainers` is the better answer for CI (a throwaway database per
session, nothing to remember to start), but it cannot give you a database to
inspect between runs. Worth adding alongside this, not instead of it.
