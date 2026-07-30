"""OpenTelemetry metrics.

A host process serving HTTP uses a Prometheus pull reader (metrics at /metrics).
A process with no HTTP server (e.g. an arq worker) uses an OTLP push reader to
send metrics to the OTel Collector, which exposes them for Prometheus to scrape.

Instrument names are prefixed with ``CoreSettings.metrics_prefix`` (default
``agentic_core``) so that two products built on this core do not collide in one
Prometheus registry. A product that has existing dashboards sets the prefix to
whatever those dashboards already query.

Instruments:
    {prefix}_runs_total               Counter   status
    {prefix}_run_duration_seconds     Histogram (no high-cardinality labels; #966)
    {prefix}_phase_duration_seconds   Histogram phase
    {prefix}_llm_calls_total          Counter   model, provider
    {prefix}_llm_call_duration_seconds Histogram model, provider
    {prefix}_llm_tokens_total         Counter   model, token_type (prompt|completion)
    {prefix}_llm_cost_dollars         Counter   model
    {prefix}_tool_calls_total         Counter   server, tool_normalized, status
    {prefix}_tool_call_duration_seconds Histogram server, tool_normalized
    {prefix}_pattern_hook_duration_seconds Histogram slug, hook_point
    {prefix}_errors_total             Counter   type, phase

Cardinality guardrails (#966, #972):
    - agent_id was removed from all histogram labels; use OTel exemplars or the
      trace_id span attribute to correlate individual runs back to histograms.
    - Tool names from MCP servers are normalized to an allowlist-based slug to
      prevent user-defined free-form strings from blowing the cardinality budget.
      Only alphanumerics + ``_`` / ``.`` / ``-`` survive; everything else is
      replaced with ``_`` and the result is truncated to 64 chars.

All instruments are module-level singletons created during setup_metrics().
Recording functions are no-ops when called before setup.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

# ---------------------------------------------------------------------------
# Cardinality helpers
# ---------------------------------------------------------------------------

_TOOL_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.:-]")
_TOOL_NAME_MAX_LEN = 64


def _normalize_tool_name(tool: str) -> str:
    """Normalize an MCP tool name to a bounded, safe label value (#972).

    Replaces any character that is not alphanumeric or one of ``_``, ``.``,
    ``:``, ``-`` with ``_``, then truncates to 64 characters.  This converts
    arbitrary user-defined tool names from MCP servers into a safe,
    bounded label set without silently discarding names.
    """
    normalized = _TOOL_NAME_SAFE_RE.sub("_", tool)
    return normalized[:_TOOL_NAME_MAX_LEN]


logger = logging.getLogger(__name__)


def _prefix() -> str:
    """Instrument-name prefix, read at ``setup_metrics`` time.

    Imported lazily so that importing this module does not materialize settings —
    ``configure()`` must still be able to install the product's instance first.
    """
    from agentic_core.config import settings

    return settings.metrics_prefix


_meter: metrics.Meter | None = None

# --- counters ---
_runs_total: metrics.Counter | None = None
_llm_calls_total: metrics.Counter | None = None
_llm_tokens_total: metrics.Counter | None = None
_llm_cost_dollars: metrics.Counter | None = None
_tool_calls_total: metrics.Counter | None = None
_errors_total: metrics.Counter | None = None

# --- histograms ---
_run_duration: metrics.Histogram | None = None
_phase_duration: metrics.Histogram | None = None
_llm_call_duration: metrics.Histogram | None = None
_tool_call_duration: metrics.Histogram | None = None
_hook_duration: metrics.Histogram | None = None


def setup_metrics(
    service_name: str = "agentic-core",
    push_endpoint: str = "",
    insecure: bool = True,
) -> None:
    """Initialise the global MeterProvider.

    Call once at application startup.

    Args:
        service_name: OTel resource service.name attribute.
        push_endpoint: When set, metrics are pushed via OTLP to this gRPC
            endpoint (e.g. ``http://otel-collector:4317``) using a
            PeriodicExportingMetricReader.  Use this for processes that have
            no HTTP server (e.g. the arq worker).  When empty, a Prometheus
            pull reader is used instead, registering metrics in the default
            prometheus_client registry so they are returned by /metrics.
        insecure: Skip TLS certificate verification when True (safe for in-cluster use).
    """
    global _meter  # noqa: PLW0603
    global _runs_total, _llm_calls_total, _llm_tokens_total, _llm_cost_dollars  # noqa: PLW0603
    global _tool_calls_total, _errors_total  # noqa: PLW0603
    global _run_duration, _phase_duration, _llm_call_duration, _tool_call_duration  # noqa: PLW0603
    global _hook_duration  # noqa: PLW0603

    resource = Resource.create({"service.name": service_name})

    readers: list[Any] = []

    if push_endpoint:
        # OTLP push — used by the worker process (no HTTP server available).
        # Only enable if the collector is actually reachable to avoid blocking
        # the event loop with endless gRPC retries.
        try:
            import grpc

            channel = grpc.insecure_channel(push_endpoint.replace("http://", ""))
            try:
                grpc.channel_ready_future(channel).result(timeout=2)
                collector_reachable = True
            except grpc.FutureTimeoutError:
                collector_reachable = False
            finally:
                channel.close()

            if not collector_reachable:
                logger.info(
                    "OTEL metrics: collector at %s not reachable, skipping push exporter",
                    push_endpoint,
                )
            else:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )
                from opentelemetry.sdk.metrics._internal.instrument import (
                    Histogram as _Histogram,
                )
                from opentelemetry.sdk.metrics.export import (
                    AggregationTemporality,
                    PeriodicExportingMetricReader,
                )

                exporter = OTLPMetricExporter(
                    endpoint=push_endpoint,
                    insecure=insecure,
                    preferred_temporality={_Histogram: AggregationTemporality.CUMULATIVE},
                )
                readers.append(PeriodicExportingMetricReader(exporter, export_interval_millis=15_000))
                logger.info("OTEL metrics: OTLP push exporter enabled → %s", push_endpoint)
        except Exception as exc:
            logger.warning("OTEL metrics: OTLP push exporter unavailable: %s", exc)
    else:
        # Prometheus pull — used by the backend FastAPI process (/metrics endpoint)
        try:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader

            readers.append(PrometheusMetricReader())
            logger.info("OTEL metrics: Prometheus exporter enabled")
        except Exception as exc:
            logger.warning("OTEL metrics: Prometheus exporter unavailable: %s", exc)

    provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(provider)
    _meter = metrics.get_meter("agentic_core", "0.1.0")

    # --- counters ---
    _runs_total = _meter.create_counter(
        f"{_prefix()}_runs_total",
        description="Total number of agent runs by status",
    )
    _llm_calls_total = _meter.create_counter(
        f"{_prefix()}_llm_calls_total",
        description="Total LLM API calls by model and provider",
    )
    _llm_tokens_total = _meter.create_counter(
        f"{_prefix()}_llm_tokens_total",
        description="Total LLM tokens consumed (prompt + completion)",
        unit="tokens",
    )
    _llm_cost_dollars = _meter.create_counter(
        f"{_prefix()}_llm_cost_dollars",
        description="Estimated LLM cost in USD",
        unit="$",
    )
    _tool_calls_total = _meter.create_counter(
        f"{_prefix()}_tool_calls_total",
        description="MCP tool calls by server, tool, and status",
    )
    _errors_total = _meter.create_counter(
        f"{_prefix()}_errors_total",
        description="Error count by type and phase",
    )

    # --- histograms ---
    _run_duration = _meter.create_histogram(
        f"{_prefix()}_run_duration_seconds",
        description="Agent run wall-clock duration",
        unit="s",
    )
    _phase_duration = _meter.create_histogram(
        f"{_prefix()}_phase_duration_seconds",
        description="Individual phase (Sense/Reason/Plan/Act) duration",
        unit="s",
    )
    _llm_call_duration = _meter.create_histogram(
        f"{_prefix()}_llm_call_duration_seconds",
        description="LLM API round-trip latency",
        unit="s",
    )
    _tool_call_duration = _meter.create_histogram(
        f"{_prefix()}_tool_call_duration_seconds",
        description="MCP tool call latency",
        unit="s",
    )
    _hook_duration = _meter.create_histogram(
        f"{_prefix()}_pattern_hook_duration_seconds",
        description="Pattern plugin hook execution duration, by plugin slug and hook point",
        unit="s",
    )


# ---------------------------------------------------------------------------
# Recording helpers — all are safe to call before setup_metrics() (no-ops).
# ---------------------------------------------------------------------------


def record_run(agent_id: str, status: str, duration_seconds: float) -> None:
    """Record a completed run (counter + histogram).

    ``agent_id`` is intentionally omitted from the histogram labels (#966):
    per-agent histograms would blow Prometheus cardinality on high-volume
    tenants.  Use OTel exemplars or the ``run_id`` span attribute to correlate
    individual run durations back to a trace.  The counter still records
    ``status`` so high-level success/failure rates remain available.
    """
    if _runs_total:
        _runs_total.add(1, {"status": status})
    if _run_duration:
        _run_duration.record(duration_seconds, {})


def record_phase(agent_id: str, phase: str, duration_seconds: float) -> None:
    """Record a completed phase.

    ``agent_id`` is dropped from phase-duration labels for the same cardinality
    reason as ``record_run``.  The ``phase`` label (sense/reason/plan/act) is
    low-cardinality and is kept so you can see which phase is the bottleneck.
    """
    if _phase_duration:
        _phase_duration.record(duration_seconds, {"phase": phase})


def record_llm_call(
    model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_seconds: float,
) -> None:
    """Record an LLM API call with token usage and cost."""
    model_attrs = {"model": model, "provider": provider}
    if _llm_calls_total:
        _llm_calls_total.add(1, model_attrs)
    if _llm_call_duration:
        _llm_call_duration.record(latency_seconds, model_attrs)
    if _llm_tokens_total:
        _llm_tokens_total.add(prompt_tokens, {"model": model, "token_type": "prompt"})
        _llm_tokens_total.add(completion_tokens, {"model": model, "token_type": "completion"})
    if _llm_cost_dollars:
        _llm_cost_dollars.add(cost_usd, {"model": model})


def record_tool_call(server: str, tool: str, success: bool, latency_seconds: float) -> None:
    """Record an MCP tool call.

    Tool names come from user-defined MCP servers and can be arbitrary
    free-form strings.  They are normalized via ``_normalize_tool_name``
    (#972) before being used as label values so the Prometheus cardinality
    budget is not blown by adversarial or overly descriptive tool names.
    """
    tool_normalized = _normalize_tool_name(tool)
    status = "success" if success else "error"
    attrs = {"server": server, "tool": tool_normalized, "status": status}
    if _tool_calls_total:
        _tool_calls_total.add(1, attrs)
    if _tool_call_duration:
        _tool_call_duration.record(latency_seconds, {"server": server, "tool": tool_normalized})


def record_hook_duration(slug: str, hook_point: str, duration_seconds: float) -> None:
    """Record a pattern-plugin hook execution duration.

    Used to validate whether a plugin's ``recommended_hook_timeout`` matches
    actual observed latency.  Labels are low-cardinality: ``slug`` is bounded by
    the set of registered plugins and ``hook_point`` is bounded by the
    ``HookPoint`` enum.
    """
    if _hook_duration:
        _hook_duration.record(duration_seconds, {"slug": slug, "hook_point": hook_point})


def record_error(error_type: str, phase: str) -> None:
    """Record an error by type and phase."""
    if _errors_total:
        _errors_total.add(1, {"type": error_type, "phase": phase})
