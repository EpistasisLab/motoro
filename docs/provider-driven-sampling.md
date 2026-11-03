# Provider discovery and sampling

Motoro resolves sampling from explicit settings and provider metadata, with no
model registry, name-based capability rules, or LiteLLM static-map fallback.
The Haiku reproduction demonstrates a local failure: unknown effort metadata was
treated as unsupported. It does **not** establish which discovery response,
cached entry, or saved capability payload reached the affected ASAREE node.
Capture that redacted payload separately when investigating the production node.

## Provider schemas verified on 2026-10-29

These are documentation/schema checks and mocked HTTP tests, not authenticated
live-provider validation. Model documentation is not a model-discovery response.

| Provider | Discovery facts used | Sampling facts not exposed by the verified listing schema |
| --- | --- | --- |
| Anthropic | Paginated `/v1/models`, ID/display name, nullable `capabilities.effort`, `supported`, named level support nodes | No default effort, temperature support/default, or temperature/effort compatibility rules |
| OpenAI | Authenticated `/v1/models`, ID (also returns creation/ownership information) | No sampling support, accepted effort ladder, defaults, or compatibility restrictions |
| OpenRouter | `/api/v1/models`, optional `supported_parameters`, `reasoning.supported_efforts`, `reasoning.default_effort`, `reasoning.mandatory`, `reasoning.default_enabled`, `default_parameters.temperature` | No temperature/effort compatibility allowlist; missing/null fields do not establish exact defaults |
| Azure Foundry | Project `/deployments?api-version=v1`, deployment `name` and underlying `modelName` | No exact sampling defaults/effort ladder or compatibility rules; `chat_completion` capability is not a sampling fact |
| Local | OpenAI-compatible `/models`, ID | No portable sampling capability/default schema; behavior varies by server |
| Bedrock | Existing inference route only; not a `ProviderConfig` discovery provider | Motoro performs no Bedrock capability discovery; no sampling facts are inferred from the model ID |

Sources: [Anthropic Models API](https://platform.claude.com/docs/en/api/models/list),
[effort SDK schema](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/effort_capability.py),
[capabilities SDK schema](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/model_capabilities.py),
[OpenAI Models API](https://developers.openai.com/api/reference/resources/models/methods/list),
[OpenRouter Models API](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties),
[OpenRouter reasoning discovery](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#discovering-per-model-reasoning-options),
[Foundry deployments](https://learn.microsoft.com/en-us/rest/api/aifoundry/project/deployments),
[Foundry REST schemas](https://learn.microsoft.com/en-us/rest/api/microsoft-foundry/aiproject),
[Ollama compatibility](https://docs.ollama.com/api/openai-compatibility).

Anthropic's [effort guide](https://platform.claude.com/docs/en/build-with-claude/effort)
and [Haiku overview](https://platform.claude.com/docs/en/models/haiku-5-5/overview)
document the five effort levels and medium default for Haiku 5.5. Those facts are
not copied into a table. Without an API-advertised default, an unset effort is
omitted so Anthropic selects it. Motoro does not report an exact API default.
The parser accepts optional default/temperature extensions **if returned**, but
tests of those extensions do not mean the standard Anthropic API exposes them.

## Shared contract and resolution

`ModelCapabilities` accepts tri-state support flags (`null` unknown), optional
defaults, optional accepted `effort_levels`, and `temperature_effort_levels`.
Missing compatibility metadata imposes no local restriction; it proves no
compatibility. Empty compatibility lists forbid temperature in every mode.

`effort_levels_complete` distinguishes an exhaustive accepted list from partial
positive evidence. It defaults to true for existing override payloads.
`unsupported_effort_levels` records explicit negative evidence. Anthropic's
independent named flags are partial positive/negative facts; missing and null
nodes remain unknown. Names are read dynamically, not from a fixed vocabulary.
OpenRouter's non-null `supported_efforts` list is exhaustive; null means a known
selector with an unknown gateway vocabulary. Mandatory reasoning excludes
explicit `none`, even with an unknown ladder. A conditional effort default does
not automatically enable reasoning when `default_enabled` is false.

`resolve_sampling(ModelInfo, temperature=..., effort=...)` consumes provenance.
Only `provider` capability/default facts constrain automatic resolution.
Catalog or missing provenance cannot become an advertised default or veto an
explicit setting. `0.0` is a real temperature; `None` means unset; effort
`"none"` is an explicit value, independent of missing effort.

Explicit values win. Known negative support, excluded efforts, exhaustive-list
violations, and known incompatible combinations reject locally. Incomplete
metadata defers explicit validation to the provider. When the effort mode is
unknown, an explicit temperature is preserved; an automatic temperature default
is omitted if compatibility cannot be established. Unset controls use provider
defaults if available and compatible; otherwise they are omitted. There is no
lowest-effort policy or temperature 0.5 fallback.

Bare `ModelCapabilities` is an explicit caller override. Defaults from that
input have source `capability_override`, not `advertised_default`. Sampling
results also expose `explicit`, `omitted`, and omission reasons. The legacy
`application_fallback` source remains deserializable but is never generated.
`get_capabilities(model, default=...)` remains importable, returns a fresh caller
default or unknown capabilities for **every** ID, and performs no I/O. `CATALOG`
remains importable as an empty deprecated compatibility surface.
The static `DEFAULT_EFFORT` and `EFFORT_LEVELS_FULL` exports are removed; audit
downstream imports and populate controls from discovered metadata instead.

Discovery is stateless and makes no inference calls. OpenAI now lists the
credential's actual models. Anthropic discovery failures/empty results no longer
return a catalog: callers receive structured errors and no fabricated facts.
An absent ID always resolves to unknown metadata. `ModelListing.source=static`
and `Provenance=catalog` remain readable for old serialized payloads, but new
discovery never generates them. Discovery exposes all returned IDs; the Models
API's inclusion of an ID does not prove that it supports chat completions.
The generic local provider does not probe server-specific APIs such as Ollama's
`/api/show`; those schemas require a dedicated server adapter. The portable
OpenAI-compatible model listing alone supplies no sampling metadata.

## Inference and compatibility

Use `ModelConfig.model_metadata=metadata` and retain the full `ModelInfo` through
`PartialModelConfig`, run configuration, and context serialization. Its ID must
match the configured model/deployment. Clear/refresh metadata when changing the
provider, endpoint, credential, or model. `resolved_capabilities` still takes
precedence as an explicit override. No inference path performs discovery.

Structured output (Instructor), text, native tools, and tool selection share
the same resolution and transport adapter. OpenAI-compatible routes use native
sampling JSON in `extra_body`; Anthropic receives `output_config.effort` and
temperature through a request-scoped HTTP overlay. This bypasses LiteLLM's
static sampling flags, token-budget guesses, and effort renaming without
mutating global maps or hooks. No thinking mode is inferred or enabled for
Anthropic: effort and thinking are independent controls. An SDK model map must
not substitute for a provider's sampling facts.

Bedrock keeps its SDK protocol adapter for Converse/Invoke and signing. A
request-scoped guard checks the final signed payload for unchanged sampling
values before sending. If that adapter converts an effort into a token budget,
renames it, or drops it, the request fails visibly; Motoro does not send a
different setting. A newer Bedrock effort may therefore require a compatible
SDK protocol adapter even though Motoro itself has no model table. This is a
transport limitation, not evidence that the model lacks the capability.
See [Converse](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
and its model-specific additional request fields.

Local configuration errors and provider bad requests are terminal across model
fallbacks. Retryable transport failures retain the original settings; explicit
fallback configurations retain their own independent metadata. Motoro never
retries with a reduced effort, a changed temperature, or a dropped control.
Applications that relied on silently omitted unsupported saved temperatures
will now see a visible error. Provider APIs can reject settings that were
previously blocked or silently transformed by Motoro/LiteLLM.

## ASAREE handoff (no ASAREE edits in this change)

1. Preserve `ModelInfo.capability_sources` and `defaults` in model-list API
   responses, capability resolution, UI previews, and persisted discovered
   metadata. Pass `model_metadata` into Motoro and the full `ModelInfo` into
   `resolve_sampling`. Do not project it down to `ModelCapabilities` and then
   re-label its defaults as provider facts.
2. Keep deliberate capability overrides in `resolved_capabilities`, separate
   from discovered `model_metadata`. Audit existing saved payloads: the same old
   field may contain either manual overrides or discovery copies. Classify by
   recorded origin, not by field presence or a model-name heuristic. Preserve
   deliberate overrides; refresh discovered copies in editable drafts. If
   provenance/origin cannot be recovered, surface that ambiguity for migration
   instead of pretending it was provider-advertised.
3. Render support as supported/unsupported/unknown. Unknown support permits
   explicit entry and explains that the provider will validate. Offer discovered
   effort names dynamically, distinguish partial suggestions from exhaustive
   ladders, and exclude explicit negative facts. Provide a provider-default/unset
   choice; display an exact default only when its provenance is known. Never
   materialize 0.5 or the first/lowest effort as the UI's automatic selection.
4. Version or invalidate the credential-scoped Redis listing cache, **including
   both its 6-hour fresh entries and 24-hour retained entries**, plus the
   30-second result cache. Old catalog-enriched/provenance-stripped payloads
   cannot be repaired by removing a single Haiku cache key. Keep credential,
   endpoint, provider, and user isolation; Motoro adds no cache of its own.
5. Preserve immutable published snapshots exactly. Do not rediscover, recompute,
   backfill, or relabel their frozen sampling on read/run. Keep the resolver and
   Motoro version used by old snapshots available for replay if behavior depended
   on old omission/translation rules. Upgrading the dependency alone cannot
   guarantee replay equivalence for those snapshots. Apply the new policy to
   drafts/new publications through a versioned migration.
6. New snapshots should record explicit input separately from discovered facts,
   overrides, resolved values/sources/omission reasons, final provider request
   sampling fields, model/deployment/provider/endpoint identity, discovery time,
   and resolver/Motoro version. Freeze the selected effective settings once.
   Do not rediscover at execution. An omitted unadvertised default is inherently
   provider-controlled and cannot promise exact future sampling reproducibility;
   record that uncertainty, and pin model versions where available. Existing
   numeric temperatures and efforts remain explicit, including zero and `none`.
7. Validate ASAREE's real API handlers and snapshot execution against this exact
   Motoro commit before a release: credential isolation, failed/stale discovery,
   unknown metadata, future names, known invalid settings, provenance round-trip,
   immutable published requests, and new publication previews. Use deterministic
   provider mocks; authenticated live smoke tests remain separate work.

Motoro requires no database migration. Product response/schema, UI, cache-version,
draft migration, and snapshot-version changes are required in ASAREE.
