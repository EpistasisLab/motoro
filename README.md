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

Feature PRs target `develop`; tested release batches are promoted to `main`.
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

### Model capabilities

`motoro.services.model_capabilities.ModelCapabilities` describes factual support,
independently of selected configuration. Its shared JSON contract is:

| Field | Meaning |
| --- | --- |
| `supports_temperature`, `supports_effort` | `true` supported, `false` unsupported, `null` unknown (default) |
| `default_temperature` | Optional advertised default, including `0.0`; `null` means unadvertised |
| `effort_levels` | Unique accepted strings, or `null` for unknown; legacy `[]` remains valid without effort support |
| `default_effort` | Optional advertised default; `null` is unset, literal `"none"` disables reasoning |
| `temperature_effort_levels` | Optional allowlist: temperature requires a resolved effort in this list; `[]` forbids every mode; `null` imposes no additional restriction |

Effort support does not require a known default or complete level list. Advertised
defaults must belong to the accepted list when that list is known. Metadata for
unsupported controls cannot advertise defaults. Unknown models now have unknown
support and omit both parameters; `get_capabilities(model, *, default=DEFAULT_CAPABILITIES)`
still accepts an explicit caller fallback. Registry matches take precedence.
Only exact model IDs and dated variants inherit registry rules, avoiding accidental
application of base-model rules to unverified variants such as `gpt-5.2-pro`.

ASAREE should validate its live capability metadata with
`motoro.services.model_capabilities.ModelCapabilities.model_validate()` and
populate `ModelConfig.resolved_capabilities` (or the same field on
`PartialModelConfig` at its agent/API boundary). Use that validated object
to render the UI and carry it into execution:

```python
from motoro.schemas.agent import ModelConfig
from motoro.services.llm_service import LLMService
from motoro.services.model_capabilities import ModelCapabilities, get_capabilities
from motoro.services.sampling import resolve_sampling

# live_metadata is supplied by the product's capability discovery.
caps = ModelCapabilities.model_validate(live_metadata)
# Without live discovery: caps = get_capabilities("my-model")
config = ModelConfig(
    provider="openai",
    model="my-model",
    resolved_capabilities=caps,
    temperature=None,  # unset, so advertised defaults remain visible
    effort=None,       # distinct from effort="none"
)
# Use the same resolver for UI previews and effective experiment settings.
effective = resolve_sampling(
    caps, temperature=config.temperature, effort=config.effort, model=config.model,
)
wire_parameters = effective.as_kwargs()  # temperature / reasoning_effort, omitting unset values
provenance = effective.model_dump()      # effective values, sources, omission reasons
text, record = await LLMService().complete_text(
    config, [{"role": "user", "content": "Hello"}]
)
```

Resolved capabilities override the registry, including conflicting known
entries. When absent, execution calls `get_capabilities()` with the effective
provider model string. The same decision applies to structured completions,
text completions, native tool calls, tool selection, and their retries.
The field survives JSON serialization and Motoro's agent/run execution path;
each `fallback_models` entry carries its own capabilities independently.
Supply metadata for the effective model, including any deployment mapping,
and refresh or clear it when changing the model or provider.

The public resolver interface is
`resolve_sampling(capabilities, *, temperature=None, effort=None, model="<unspecified>") -> ResolvedSampling`.
It has no dependency on product discovery, databases, or execution schemas.
`ResolvedSampling` contains `temperature`, `effort`, `temperature_source`,
`effort_source`, and separate `*_omission_reason` fields. Sources are `explicit`,
`advertised_default`, `application_fallback`, and `omitted`.

Temperature resolves to explicit configuration, then the advertised default, then
application fallback `0.5`. Effort resolves to explicit configuration, then the
advertised default (including `"none"`), then the lowest accepted active level in
the semantic order `minimal < low < medium < high < xhigh < max`. Provider list
order never determines policy. Disabled `"none"` is never an automatic fallback.
Unranked provider-specific strings can be selected explicitly or advertised as
defaults, but are not guessed as fallbacks. Unknown accepted levels omit effort,
unless an advertised default itself establishes an accepted value. Other explicit
values cannot be validated against an unknown list and fail visibly.

Both controls are sent when supported and compatible. A combination allowlist
is checked against resolved effort. An explicit incompatible temperature raises
`SamplingCombinationError`; an unset temperature in that mode stays omitted with
reason `incompatible_effort`, without generating a temperature default. Unknown
or unsupported temperature support omits even legacy saved temperatures.
Unsupported explicit efforts raise `UnsupportedEffortError`, including on
temperature-only models. Both errors subclass `SamplingConfigurationError`
(`ValueError`) and terminate without retry or fallback. `UnsupportedEffortError`
also remains importable from `llm_service` for existing callers.

For resolved OpenRouter requests only, a validated effective effort still adds
`allowed_openai_params=["reasoning_effort"]`, including when temperature is also
sent. Omitted effort gets no allowance. Motoro does not enable global `drop_params`.

For custom temperature-only models, pass
`resolved_capabilities=TEMPERATURE_ONLY_CAPABILITIES`, imported from
`motoro.services.model_capabilities`, or equivalent metadata:

```python
{"supports_temperature": True, "supports_effort": False,
 "effort_levels": [], "default_effort": None}
```

Compatibility: `ModelConfig.temperature` and `PartialModelConfig.temperature`
now default to `None` and accept JSON `null`. Existing saved numeric values,
including `0.7` and `0.0`, remain explicit through serialization and execution.
Old capability payloads with both boolean flags, levels, and a default still
validate. Previously fabricated unknown-model medium effort is no longer sent.
Products should preserve missing defaults as `null` and call the shared resolver
instead of copying its policy. No database migration is required in Motoro.

Provider rules verified against official documentation on 2026-10-27:
[OpenAI compatibility](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2)
states original GPT-5/mini/nano reject temperature, while GPT-5.1/5.2 accept it
only at effort `"none"`. The registry represents both controls for 5.1/5.2 with
`temperature_effort_levels=["none"]`, and their advertised default is `"none"`
([GPT-5.1](https://developers.openai.com/api/docs/models/gpt-5.1),
[GPT-5.2](https://developers.openai.com/api/docs/models/gpt-5.2)). Original
[GPT-5](https://developers.openai.com/api/docs/models/gpt-5) includes `minimal`.
The existing curated Claude effort-only entries now record the advertised `high`
default rather than the former suggested `medium`
([Anthropic effort](https://platform.claude.com/docs/en/build-with-claude/effort)).
Live provider/deployment metadata takes precedence over these curated entries;
the registry does not discover or infer other thinking modes.

### Provider checks, listings, and defaults

Use `motoro.services.providers` without a database, runtime, cache, or product
models. Configuration is explicit; these operations never consult environment
credentials or the inference credential resolver. The caller owns authorization,
credential storage, rate limits, and cache policy.

```python
from motoro.services.provider_types import ProviderConfig
from motoro.services.providers import (
    check_connection, discover_models, get_model_metadata, get_model_defaults,
)

# Within an async function. OpenRouter listings are also public without a key.
config = ProviderConfig(provider="openrouter", api_key="your-key")
check = await check_connection(config)
listing = await discover_models(config)
metadata = get_model_metadata(model="vendor/model", listing=listing)
defaults = get_model_defaults(model="vendor/model", listing=listing)
if defaults.temperature.known:
    print(defaults.temperature.value, defaults.temperature.source)
```

The runnable [independent example](examples/provider_metadata.py) uses
`httpx.MockTransport` and makes no network requests. Both async operations accept
`transport: httpx.AsyncBaseTransport | None = None` for deterministic testing.
For connection pooling across operations, pass `client: httpx.AsyncClient` instead:

```python
import httpx

async with httpx.AsyncClient() as client:
    check = await check_connection(config, client=client)
    listing = await discover_models(config, client=client)
```

The caller owns and closes the supplied client. `client` and `transport` are
mutually exclusive. Each request overrides the client's timeout, redirect policy,
and default authentication; credentials come from `ProviderConfig`. Use a dedicated
client without unrelated default headers, cookies, or query parameters.
Only explicit connection checks and discovery issue requests, all GETs with a
10-second timeout and no redirects or automatic retries. Metadata/default
lookups are synchronous and perform no I/O. No metadata operation runs inference.

`ProviderConfig` accepts `provider`, optional secret `api_key`, `api_base`, and
`azure_project_endpoint`. Supported providers are `anthropic`, `openai`,
`azure_foundry`, `openrouter`, and `local`. Keys are excluded from serialization
and repr. Endpoints must have valid ports and cannot contain URL credentials,
queries, or fragments. Local bases include the server's API path, typically
`/v1`; Azure deployment discovery
requires the separately supplied project endpoint, not just the inference host.

`discover_models(config)` returns `ModelListing(models, source, note, error)`.
Each `ModelInfo` has `id`, `label`, existing `ModelCapabilities`, per-field
`capability_sources`, and `ModelDefaults`. OpenAI uses the curated catalog;
Anthropic uses paginated live metadata and falls back to the catalog on empty
or failed discovery. OpenRouter uses its public live catalog, optionally with
credentials. Azure uses project deployments and resolves sampling metadata by
underlying model name. Local uses best-effort OpenAI-compatible `/models`.

The listing source remains `api`, `static`, or `error`, so callers can preserve
existing cache/fallback policy. Empty Azure deployments are a successful `api`
listing; malformed or missing list fields remain errors. Empty local/OpenRouter
listings are `error`; empty Anthropic listings fall back to `static`.
A structured `ProviderError(kind, http_status)` retains
credential rejection, rate limiting, timeout, network, HTTP, malformed response,
missing credentials, unavailable discovery, and empty-list outcomes, including
when a catalog fallback succeeds. Error text never includes raw response bodies
or provider exception strings. A caller can retain `error` and `note` alongside
a cached listing; Motoro implements no cache or stale-fallback policy.

`check_connection(config)` returns `ConnectionCheck(status, detail, endpoint,
models, error)`, with `ok`, `failed`, or `unknown`. Accepted keys do not establish
quota or billing readiness. Direct Anthropic/OpenAI/OpenRouter rate limits are
inconclusive. Azure and local reuse discovery: unavailable Azure project
configuration and unsupported local listings are inconclusive; Azure discovery
failures fail the check. Successful Azure checks return the exact deployments
used for the check so a caller can seed its own cache.

`get_model_metadata(model=..., listing=...)` prefers a matching listing entry,
then the existing registry. Unknown model fields remain `None`. Check the
supplied listing's `error`/`note` separately: metadata lookup does not erase its
failure or establish whether that model is accessible to the credential.
`get_model_defaults` retains explicitly advertised defaults independently of
sampling support metadata, using curated capability defaults as a fallback.
Omitted defaults fall back to the catalog even when a live capability ladder
excludes that default; an explicitly advertised `null` remains unknown.
An advertised default does not establish support or change inference settings.
The shared fields are `temperature` and `effort`; other advertised defaults are in
`additional`. Each `DefaultValue` exposes `value`, `source` (`provider`,
`catalog`, or `unknown`), and the Python property `known`. In serialized JSON,
`value: null, source: unknown` denotes unknown; `0`, `false`, `""`, `[]`, and
`{}` remain known values. Limits such as context size are not invented output
token defaults. Application preferences and overrides still belong in
`ModelConfig` and `resolve_sampling`, separately from these facts.

For inference adapters, `get_inference_connection(config, model=...)` returns
the existing `motoro.services.credentials.Connection` dictionary, including
Foundry endpoint normalization and the Azure/local LiteLLM routing prefixes.
Return it from a caller-owned credential resolver; it deliberately contains
the secret key and must never be serialized to browser responses or logs.
Pass `metadata.capabilities` as `ModelConfig.resolved_capabilities` to use the
same provider facts during inference. For tri-state tool support, use
`motoro.services.llm_service.model_tool_calling_support(provider, model)`;
unknown models return `None`. The existing execution-pattern boolean helper
continues to treat unknown support conservatively as false.

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
