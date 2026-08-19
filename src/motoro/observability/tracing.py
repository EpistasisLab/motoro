"""OpenTelemetry distributed tracing.

Each agent run is a trace; each phase (Sense/Reason/Plan/Act) is a child span.
LLM calls and MCP tool calls are nested spans within their parent phase.

Traces are exported via OTLP gRPC to the configured collector endpoint.
Gracefully degrades to a no-op provider if the collector is unavailable.

Sampling strategy (#989)
------------------------
``setup_tracing`` uses ``ErrorAwareSampler``, which wraps a ``TraceIdRatioBased``
sampler with "force-keep on errors" semantics:

- Spans that *would* be dropped by the ratio sampler are downgraded to
  ``RECORD_ONLY`` (recorded in memory but not exported) so that if the span
  later terminates with an error status the ``ErrorForwardingSpanProcessor`` can
  still export it.
- ``ErrorForwardingSpanProcessor`` intercepts ``on_end()`` and forwards any span
  whose OTel status is ERROR directly to the underlying exporter, even if it was
  originally marked ``RECORD_ONLY`` by the sampler.

This gives head-based ratio sampling for normal traces (low volume) while
ensuring every error trace is exported regardless of the sample rate.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import (
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
    TraceIdRatioBased,
)
from opentelemetry.trace import Link, SpanKind
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.types import Attributes

logger = logging.getLogger(__name__)

_provider: TracerProvider | None = None


class ErrorAwareSampler(Sampler):
    """Head-based ratio sampler that downgrades DROP to RECORD_ONLY (#989).

    The downgrade ensures spans are in memory so ``ErrorForwardingSpanProcessor``
    can force-export any span that ends with OTel status ERROR, even when the
    random trace-ID ratio would normally drop it.

    At sample_rate=1.0 this is equivalent to ``ALWAYS_ON``.
    """

    def __init__(self, sample_rate: float) -> None:
        self._inner = ParentBased(root=TraceIdRatioBased(sample_rate))

    def should_sample(
        self,
        parent_context: otel_context.Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes | None = None,
        links: Sequence[Link] | None = None,
        trace_state: Any | None = None,
    ) -> SamplingResult:
        result = self._inner.should_sample(parent_context, trace_id, name, kind, attributes, links, trace_state)
        if result.decision == Decision.DROP:
            # Downgrade: record in memory so error spans can still be exported.
            return SamplingResult(
                decision=Decision.RECORD_ONLY,
                attributes=result.attributes,
                trace_state=result.trace_state,
            )
        return result

    def get_description(self) -> str:
        return f"ErrorAwareSampler({self._inner.get_description()})"  # type: ignore[no-untyped-call]


class ErrorForwardingSpanProcessor(BatchSpanProcessor):
    """BatchSpanProcessor that force-exports RECORD_ONLY spans with ERROR status (#989).

    Spans that were sampled out (RECORD_ONLY) but end with OTel StatusCode.ERROR
    are forwarded synchronously to the exporter so no error trace is lost even
    at low sampling ratios.
    """

    def __init__(self, exporter: SpanExporter, **kwargs: Any) -> None:
        super().__init__(exporter, **kwargs)
        self._error_exporter = exporter
        self._error_export_lock = threading.Lock()

    def on_end(self, span: ReadableSpan) -> None:
        # Always let the normal BatchSpanProcessor handle SAMPLED spans.
        super().on_end(span)

        # Additionally force-export RECORD_ONLY spans whose status is ERROR.
        ctx = span.context
        if ctx is None:
            return

        sampled = bool(ctx.trace_flags & 0x01)
        if not sampled and span.status is not None and span.status.status_code == StatusCode.ERROR:
            # Export the single error span synchronously (small overhead,
            # high diagnostic value).
            with self._error_export_lock:
                try:
                    self._error_exporter.export([span])
                except Exception:
                    logger.debug("ErrorForwardingSpanProcessor: export failed for error span")


def setup_tracing(
    service_name: str = "motoro",
    otlp_endpoint: str = "",
    sample_rate: float = 1.0,
    insecure: bool = True,
) -> None:
    """Initialise the global TracerProvider.

    Call once at application startup before any get_tracer() calls.

    Args:
        service_name: OpenTelemetry ``service.name`` resource attribute.
        otlp_endpoint: OTLP gRPC endpoint (e.g. ``http://otel-collector:4317``).
            When empty no span exporter is attached (silent no-op for dev).
        sample_rate: Fraction of traces to sample (0.0–1.0).  1.0 = 100 %.
        insecure: When True, skip TLS certificate verification for the OTLP exporter.
            Safe for in-cluster collectors; set False for internet-facing endpoints.
            Read from ``settings.otel_exporter_otlp_insecure`` at call sites (#990).
    """
    global _provider  # noqa: PLW0603

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",
            "deployment.environment": "production" if sample_rate < 1.0 else "development",
        }
    )

    # #989: ErrorAwareSampler keeps RECORD_ONLY for ratio-dropped spans so
    # ErrorForwardingSpanProcessor can still export them when they end with ERROR.
    sampler: Sampler = ErrorAwareSampler(sample_rate)
    provider = TracerProvider(resource=resource, sampler=sampler)

    if otlp_endpoint:
        exporter = _build_otlp_exporter(otlp_endpoint, insecure=insecure)
        if exporter is not None:
            provider.add_span_processor(ErrorForwardingSpanProcessor(exporter))
            logger.info(
                "OTEL tracing: OTLP exporter → %s (sample_rate=%.2f, error_force_keep=True)",
                otlp_endpoint,
                sample_rate,
            )
        else:
            logger.warning("OTEL tracing: OTLP exporter unavailable, traces will not be exported")
    else:
        logger.debug("OTEL tracing: no OTLP endpoint configured, spans are no-ops")

    trace.set_tracer_provider(provider)
    _provider = provider


def _build_otlp_exporter(endpoint: str, insecure: bool = True) -> SpanExporter | None:
    """Construct an OTLP gRPC span exporter.  Returns None on import/config error.

    Args:
        endpoint: OTLP gRPC endpoint URL.
        insecure: Skip TLS certificate verification when True (safe for in-cluster use).
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
    except Exception as exc:
        logger.warning("Failed to build OTLP span exporter: %s", exc)
        return None


def get_tracer(component: str = "motoro") -> trace.Tracer:
    """Return a tracer scoped to *component*.  Safe to call before setup_tracing()."""
    return trace.get_tracer(f"motoro.{component}")


def get_trace_context() -> dict[str, str]:
    """Return the active trace_id and span_id as hex strings.

    Returns empty strings when there is no active span (e.g. outside a trace).
    Used by the structlog processor to correlate logs with traces.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return {"trace_id": "", "span_id": ""}
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
    }


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the provider.  Call on app shutdown."""
    if _provider is not None:
        _provider.shutdown()


# Convenience: a pre-built dict of span attributes for common use-sites
def span_attrs(**kwargs: Any) -> dict[str, Any]:
    """Build a span-attributes dict, dropping None values."""
    return {k: v for k, v in kwargs.items() if v is not None}
