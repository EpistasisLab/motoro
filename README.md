# Motoro

A **headless agentic platform**: the Sense-Reason-Plan-Act runtime, the pattern
engine, the LLM bridge, MCP integration, memory, and the persistence beneath
them.

Core ships **no `FastAPI` application, no routers, and no UI**. A product owns its
HTTP layer, its frontend, and its own domain tables, and depends on core for the
machinery. `ASAREE` builds on it today; `ecoxai` is planned.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architectural boundary and the
rationale behind it — why core has no user model, why it owns a separate
database, how patterns and memory and MCP registration are designed, and the
real bugs that shaped each of those decisions.

## Layout

```
src/motoro/
  config.py           CoreSettings + configure() + the settings proxy
  runner.py           create_agent/create_run/execute_run/fail_run — the public lifecycle
  models/             SQLAlchemy models; models/base.py owns the shared Base
  schemas/            Pydantic contracts
  services/           LLM bridge, cost, retry, scrubbing, memory service, MCP registration, encryption
  mcp/                MCP client, registry, tool adapters (transport only)
  mcp_servers/        bundled, ready-to-register MCP servers that ship with core itself
  memory/             working (Redis), episodic + semantic (pgvector), embedding
  engine/             the SRPA loop, and later the pattern engine
  migrations/         Alembic environment + versions for core's own database
  worker/             long-running worker helpers (e.g. DB connection resilience)
  observability/      tracing + metrics
  security/           prompt-injection fencing (isolation is undecided), MCP command allowlist, SSRF guard
tests/
```

## Installing

```toml
[project]
dependencies = ["motoro"]

[tool.uv.sources]
motoro = { git = "https://github.com/EpistasisLab/motoro.git", tag = "v0.2.0" }
```

Pin by tag (semver, against core's public surface), not a raw commit SHA. To
cut a new release: land the change on `main`, confirm dependent products still
work against it, then `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`.

That tag *is* the version: `hatch-vcs` derives both the package version and
`motoro.__version__` from it at build time, so there is no version string to
edit here and none to forget. Tagging is the whole procedure.

```python
from pydantic_settings import SettingsConfigDict
from motoro import CoreSettings, configure

class Settings(CoreSettings):
    model_config = SettingsConfigDict(env_prefix="ASAREE_", env_file=".env", extra="ignore")
    # product-only fields here

configure(Settings())   # before any core module reads a setting
```

Core's only package-root export is lifecycle; everything else is imported from
its own module (settings, MCP client, patterns, memory, ...) — see
`docs/DESIGN.md` for why there's no facade.

## Using core in a product

A product does not run a script or shell out to a CLI — it imports the same
functions directly into wherever its own request handling already lives (a
FastAPI router, a Django view, a worker task, anything). This is the actual
integration shape; the `examples/` CLI walkthrough further down
("Running the examples") is for trying core out, not for a real product to
run.

```python
from motoro.runner import create_agent, create_run, execute_run
from motoro.mcp.registry import get_registry

# e.g. behind POST /agents — once; agents are durable resources
agent = await create_agent(
    name="my-agent", goal="...", system_prompt="...",
    model_config={"provider": "anthropic", "model": "claude-sonnet-5"},
    owner_id=user.id,
)

# e.g. behind POST /runs — per request
run = await create_run(agent_id=agent.id, user_input=user_input, owner_id=user.id)
await execute_run(run_id=run.id, registry=get_registry(), available_tools=[...])
```

`create_run` and `execute_run` are separate calls so a product *can* enqueue
execution instead of blocking on it (create the run in the request, execute it
in a worker) — calling both inline in the same request, like above, is just
as valid; core has no opinion on which a product picks.

```python
from motoro.runner import fail_run

# e.g. a stale-run sweep keyed on AgentRun.last_heartbeat_at
await fail_run(run_id=run.id, error="worker died mid-flight")
```

`execute_run`'s own commit is the only thing that normally moves a run out of
a non-terminal status. `fail_run` is the way in from outside that loop — for a
run whose process died (killed worker, crashed host) with nothing left to
close it out. It's a no-op on a run that's already COMPLETED, FAILED or
CANCELLED, so a detector racing a slow-but-live `execute_run` can't clobber a
status it already wrote.

## Requirements

Core requires **PostgreSQL and Redis**. A product provisions both, points
`database_url` at a database for core, and applies core's own schema as a
deploy step:

```bash
python -m motoro.migrations upgrade --url "$MOTORO_DATABASE_URL"   # core's DB
alembic upgrade head                                                     # the product's own DB
```

The two databases and their migration chains are independent — no ordering
constraint, no shared tables.

### Setting a provider

```bash
ANTHROPIC_API_KEY=sk-ant-...
# or, for Anthropic on Microsoft Foundry — BOTH are required:
ANTHROPIC_FOUNDRY_API_KEY=...
ANTHROPIC_FOUNDRY_RESOURCE=my-resource
# or OpenRouter:
OPENROUTER_API_KEY=sk-or-...
# or a self-hosted OpenAI-compatible server (LM Studio, vLLM, llama.cpp, …) —
# LOCAL_LLM_API_BASE is required (there's no default host); LOCAL_LLM_API_KEY
# is optional since most such servers don't check one:
LOCAL_LLM_API_BASE=http://localhost:8000/v1
```

These are bare-name conventional variables, not prefixed by the product's own
settings prefix. A product with its own credential store installs a resolver:

```python
from motoro.services.credentials import set_credential_resolver
set_credential_resolver(my_resolver)   # e.g. ASAREE reads a per-user settings table
```

### Model capabilities

Not every model accepts `temperature`. Adaptive-thinking models (Opus 4.7/4.8,
Sonnet 5, Fable 5) 400 on an explicit `temperature` and are steered instead
with an `effort` dial (`low | medium | high | xhigh | max`). `model_config`
doesn't need to know which regime a model is in — the LLM bridge resolves it
per call via `motoro.services.model_capabilities`, the single source of
truth other products (and their own GUI/SDK) resolve against too.

### Timeouts

`llm_call_timeout_seconds` (default 120) caps a single completion attempt, and
`hook_timeout_seconds` (default 30) wraps one pattern-hook invocation around
it — both are deployment-dependent, not a core concern, and both are plain
`CoreSettings` fields a product overrides in its own subclass. Keep
`hook_timeout_seconds` comfortably above `llm_call_timeout_seconds`: the abort
error always names `hook_timeout_seconds` regardless of which of the two
actually fired, so a run failing with a "Hook '...' timed out after 30s"
message is often really the inner LLM-call timeout being too tight for a
slow provider or a high reasoning-effort completion.

## Development

```bash
cp .env.example .env
docker compose up -d                    # postgres :5453, redis :6381, schema applied

python -m venv .venv && .venv/bin/pip install -e '.[dev]'
set -a && . ./.env && set +a

.venv/bin/pytest
.venv/bin/ruff check src tests scripts examples
.venv/bin/mypy src
.venv/bin/lint-imports                  # core must never import a product
```

### Running the examples

`examples/` walks the same functions shown above from a bare CLI, laid out as
the three phases a product genuinely has — useful for seeing core work end to
end with no HTTP app around it, not something a real product runs.

```bash
# 1. DEPLOY STEP — once per release, before the app starts
docker compose up -d                    # the motoro-migrate service does this
# ...or, without Compose:
python -m motoro.migrations upgrade --url "$MOTORO_DATABASE_URL"

# 2. PROVISIONING — once; agents are durable resources
python examples/provision.py            # prints the agent id; idempotent by name

# 3. RUNTIME — per request
python examples/run.py --agent-id <id> --input "What is 17 * 23?"

# 3b. RUNTIME, with episodic memory
python examples/memory_run.py --input "My favourite programming language is Rust."
python examples/memory_run.py --input "What did I say my favourite language was?"

# 3c. RUNTIME, with a real MCP tool call
python examples/mcp_run.py --input "What is the secret code for alpha?" --trace
```

`examples/settings.py` holds the one thing all three share: a `CoreSettings`
subclass with the product's env prefix. See `docs/DESIGN.md` for why
migrations are a deploy step rather than run at app startup.
