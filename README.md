# agentic-core

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
src/agentic_core/
  config.py           CoreSettings + configure() + the settings proxy
  models/             SQLAlchemy models; models/base.py owns the shared Base
  schemas/            Pydantic contracts
  services/           LLM bridge, cost, retry, scrubbing, memory service, MCP registration, encryption
  mcp/                MCP client, registry, tool adapters (transport only)
  memory/             working (Redis), episodic + semantic (pgvector), embedding
  engine/             the SRPA loop, and later the pattern engine
  observability/      tracing + metrics
  security/           prompt-injection fencing (isolation is undecided), MCP command allowlist, SSRF guard
tests/
```

## Installing

```toml
[project]
dependencies = ["agentic-core"]

[tool.uv.sources]
agentic-core = { git = "https://github.com/EpistasisLab/agentic-core.git", tag = "v0.1.0" }
```

Pin by tag (semver, against core's public surface), not a raw commit SHA. To
cut a new release: land the change on `main`, confirm dependent products still
work against it, then `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`.

```python
from pydantic_settings import SettingsConfigDict
from agentic_core import CoreSettings, configure

class Settings(CoreSettings):
    model_config = SettingsConfigDict(env_prefix="ASAREE_", env_file=".env", extra="ignore")
    # product-only fields here

configure(Settings())   # before any core module reads a setting
```

Core's only package-root export is lifecycle; everything else is imported from
its own module (settings, MCP client, patterns, memory, ...) — see
`docs/DESIGN.md` for why there's no facade.

## Requirements

Core requires **PostgreSQL and Redis**. A product provisions both, points
`database_url` at a database for core, and applies core's own schema as a
deploy step:

```bash
python -m agentic_core.migrations upgrade --url "$AGENTIC_DATABASE_URL"   # core's DB
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
```

These are bare-name conventional variables, not prefixed by the product's own
settings prefix. A product with its own credential store installs a resolver:

```python
from agentic_core.services.credentials import set_credential_resolver
set_credential_resolver(my_resolver)   # e.g. ASAREE reads a per-user settings table
```

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

### How a product uses core — three phases

`examples/` is laid out as the three phases a product genuinely has; only the
third belongs on a request path.

```bash
# 1. DEPLOY STEP — once per release, before the app starts
docker compose up -d                    # the agentic-core-migrate service does this
# ...or, without Compose:
python -m agentic_core.migrations upgrade --url "$AGENTIC_DATABASE_URL"

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
migrations are a deploy step rather than run at app startup, and how
`create_run`/`execute_run` being separate calls lets a product enqueue
execution instead of blocking on it.
