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

## Features

| Area | What core gives you |
|---|---|
| **Agent runtime** | Sense → Reason → Plan → Act over durable agents and runs |
| **Pattern engine** | Plugin patterns that hook the phase boundaries |
| **LLM bridge** | Structured output across five provider paths |
| **Output contracts** | A universal envelope, optionally extracted into typed fields |
| **Memory** | Working, episodic and semantic behind one service |
| **MCP** | Client, registry, persisted servers, bundled servers |
| **Agent Skills** | `SKILL.md` with three-level progressive disclosure |
| **Security** | Injection fencing, SSRF guard, allowlists, encryption at rest |
| **Observability** | OTel traces and metrics, per-model cost tracking |
| **Persistence** | Core's own Postgres schema and migration chain |

**Agent runtime**

- `create_agent` provisions once — agents are durable resources
- `create_run` and `execute_run` are separate, so a product can run inline or
  hand execution to a worker
- `fail_run` closes out a run whose process died; it's a no-op on a run already
  in a terminal status, so a stale-run sweep can't clobber a live loop
- Runs carry heartbeats for that sweep to key on

**Pattern engine**

- Patterns are plugin classes registered by slug — `reason_act` and
  `single_agent_baseline` ship built in
- Composition resolves hook ordering, detects conflicts, validates dependencies
- The class is the source of truth for metadata, so `pattern_config` is
  validated and `pattern_params` defaulted at agent-creation time — not after
  the model has been billed

**LLM bridge**

- Multi-provider structured output via litellm + Instructor
- Anthropic (direct, Microsoft Foundry, or Bedrock), OpenAI, OpenRouter, and
  any self-hosted OpenAI-compatible server
- `model_capabilities` describes independent temperature and effort support;
  `sampling.resolve_sampling` selects and validates both controls
- Structured-output retries, per-call timeouts, equal-jitter backoff

**Output contracts**

- Every run yields a universal envelope, assembled from run data with no extra
  LLM call
- An agent declaring an `output_contract` also gets one extraction pass into
  typed fields
- Every contracted field is optional and inference is forbidden — a failed
  extraction degrades to `None` with a caveat rather than killing the run

**Memory**

- **Working** — Redis, intra-run, sliding-window or LLM-summarization eviction,
  guarded by a per-run lock so concurrent writers can't double-summarize
- **Episodic** — per-run summaries that carry learning across runs
- **Semantic** — pgvector, embedding-based recall
- All three behind one `MemoryService`

**MCP**

- Client over stdio, HTTP and SSE; registry for concurrent connections; tool
  adapters that expose MCP tools to the Act phase
- Server configs persist — every process after the first reconnects from the
  table, not from code
- Subprocess environments are allowlisted, so credentials don't leak into
  untrusted servers
- Bundled and ready to register: an OKF (Open Knowledge Format) server that
  mediates a knowledge bundle instead of exposing the filesystem — reads return
  pre-parsed, filtered JSON; writes enforce the spec's structural rules

**Agent Skills**

- `SKILL.md` documents with YAML frontmatter, stored as rows
- Three disclosure levels honoured end to end: the index always in context, the
  body loaded on request as a tool result, bundled reference files read only
  when the body points at one
- A skill can be ingested as its whole directory, not just its `SKILL.md`

**Security**

- Prompt-injection fencing that delimits user-controlled text as data
- SSRF guard for every user-supplied outbound URL
- MCP command allowlist
- Fernet encryption for secrets at rest
- Credential scrubbing over error messages and logs

**Observability**

- OTel tracing: a run is a trace; each phase, LLM call and tool call is a span
- Error-aware sampling force-keeps failed spans the ratio sampler would drop
- Metrics via Prometheus pull or OTLP push, under a configurable prefix so two
  products don't collide in one registry
- Per-model cost tracking with configurable overrides and a litellm fallback

**Persistence**

- Core owns its own PostgreSQL database and Alembic chain, independent of the
  product's
- `python -m motoro.migrations upgrade` is a deploy step; every migration is
  safe to re-run over an existing schema
- Worker helpers keep long-running processes resilient to dropped connections

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

Pin by tag (semver, against core's public surface), not a raw commit SHA.

`main` is the default branch and contains the latest released code. Feature PRs
explicitly target `develop`; tested release batches are promoted to `main`.
See [Contributing and releasing](CONTRIBUTING.md) for branch roles, protections,
merge methods, dependent-product validation, tagging, and hotfixes.

The release tag defines the version: `hatch-vcs` derives both the package version
and `motoro.__version__` from it at build time.

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

### Per-run agent configuration

Use `config_snapshot` to run concurrent variants of one agent. It accepts a
dict, `AgentSnapshotUpdate`, or `AgentUpdate`, with the same public field names
as the agent schemas. Snapshot validation accepts empty names/goals and limit
values allowed in stored agents; model and pattern validation remain unchanged.
Omitted fields are captured from the agent when the run is created; each
supplied field replaces that whole setting. Optional fields can
be explicitly cleared with `None`.

```python
run = await create_run(
    agent_id=agent.id,
    user_input=user_input,
    config_snapshot={
        "system_prompt": "Review the evidence carefully.",
        "model_config": {"provider": "openai", "model": "gpt-4o", "max_tokens": 2048},
        "pattern_config": {
            "execution_pattern": "reason_act",
            "pattern_params": {"reason_act": {"max_iterations": 6}},
        },
        "skill_config": {"skill_ids": [str(skill.id)]},
        "output_contract": {"fields": [{"name": "approved", "type": "bool"}]},
    },
)
```

Snapshots cover name, goal, description, system prompt, model, pattern, skills,
output contract, memory, budget and duration. API keys are excluded from
snapshots; credentials still resolve per call. Skill IDs are pinned, while
skill bodies resolve at execution time. Tools remain per execution through
`execute_run(available_tools=...)`.

`model_config_overrides` still shallow-merges on top of the snapshot. A pinned
pattern takes precedence over `pattern_overrides`; new callers should put their
pattern in the snapshot. ReAct telemetry is stored in
`run.run_metadata["reason_act_state"]`, preserving other metadata, instead of
writing runtime state into `pattern_overrides`.

Without a snapshot, the run reads the agent's settings when execution starts,
then records them in `run.config_snapshot` for audit and subsequent executions.
`motoro.runner.effective_agent_config(run)` centralizes this resolution and
returns a validated `AgentExecutionConfig` from `motoro.schemas.run_config`,
which accepts stored agent values without create-request constraints. Budget
and duration remain product-enforced settings; the helper exposes their pinned
values without requiring a product
to reread the shared agent.

### Same-named MCP tools

Pass the plain catalog from `registry.get_all_tools()` as `available_tools`.
When two granted servers expose the same bare name, such as `ping`, Motoro
derives unique provider function names from `server.tool` and maps calls back
to the appropriate server. Tools with unique bare names keep those readable
provider names. Text-based selection uses namespaced names for collisions too;
planned steps must use the namespaced names shown in the prompt catalog when
the bare name collides. Plan validation clears ambiguous bare names so Act
can select an available tool explicitly.

Descriptors must retain `server_id`, `server`, `tool_name` and `name`. Legacy
descriptors whose `tool_name` was rewritten to `server.tool` are also accepted.
Dispatch still checks the resolved server and tool against the run's grants.

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

### Model capabilities and provider discovery

Sampling uses provider facts without a per-model registry or LiteLLM model-map
fallback. Discovery is stateless: callers supply credentials and own caching.

```python
from motoro.schemas.agent import ModelConfig
from motoro.services.provider_types import ProviderConfig
from motoro.services.providers import discover_models, get_model_metadata
from motoro.services.sampling import resolve_sampling

listing = await discover_models(ProviderConfig(provider="anthropic", api_key="your-key"))
metadata = get_model_metadata(model="claude-haiku-5-5", listing=listing)
config = ModelConfig(provider="anthropic", model=metadata.id,
                     model_metadata=metadata, effort="medium")
effective = resolve_sampling(metadata, temperature=config.temperature, effort=config.effort)
```

Retain the entire `ModelInfo`, including `capability_sources` and `defaults`,
in `ModelConfig.model_metadata` (also available on `PartialModelConfig`).
Only facts with provider provenance drive automatic resolution. Explicit
`resolved_capabilities` remains a caller override and takes precedence;
its defaults are labelled `capability_override`, not `advertised_default`.

Explicit temperature/effort is preserved unless contradicted by known facts.
Unknown support or incomplete effort ladders defer validation to the provider.
Unset controls use advertised defaults when available; otherwise they are omitted.
There is no lowest-effort or temperature 0.5 fallback. Known incompatible
combinations raise `SamplingCombinationError`; known unsupported efforts raise
`UnsupportedEffortError`. Both are terminal `SamplingConfigurationError` values.
Provider bad requests also terminate without trying a different fallback configuration.

`discover_models` and `check_connection` accept a caller-owned
`httpx.AsyncClient` or `httpx.MockTransport`; requests use a 10-second timeout,
explicit credentials, and no redirects or automatic retries. Model/default
lookups perform no I/O. All five discovery providers use live APIs; failed
discovery returns an empty error listing. Azure needs a project endpoint.
Model IDs can still be supplied explicitly for inference.

See [the provider schema notes and ASAREE migration handoff](docs/provider-driven-sampling.md)
for field semantics, API limitations, provenance, wire behavior, caching, and
published snapshot compatibility. The offline [provider example](examples/provider_metadata.py)
uses deterministic mock responses.

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
