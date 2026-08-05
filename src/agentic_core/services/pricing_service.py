"""Pricing service — configurable per-model cost overrides with litellm fallback."""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from decimal import Decimal

import litellm
import structlog
from litellm.exceptions import NotFoundError as LiteLLMNotFoundError
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_core.models.pricing import LLMPricingOverride
from agentic_core.schemas.pricing import PricingOverrideCreate, PricingOverrideList, PricingOverrideResponse

log = structlog.get_logger()

_CACHE_MAX_SIZE = 500
_cache: OrderedDict[str, tuple[Decimal, Decimal]] = OrderedDict()
_cache_lock = asyncio.Lock()


async def refresh_cache(db: AsyncSession) -> None:
    """Reload pricing overrides from the database into the in-memory cache."""
    result = await db.execute(select(LLMPricingOverride))
    overrides = result.scalars().all()
    async with _cache_lock:
        _cache.clear()
        for o in overrides:
            _cache[o.model_name] = (o.input_cost_per_mtok, o.output_cost_per_mtok)
        # Evict oldest entries if cache exceeds max size
        while len(_cache) > _CACHE_MAX_SIZE:
            _cache.popitem(last=False)
    log.debug("pricing.cache_refreshed", count=len(_cache))


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    completion_response: object | None = None,
) -> tuple[float | None, str]:
    """Calculate cost for an LLM call.

    Returns (cost_usd, source) where source is 'override' or 'litellm'.
    Returns (None, 'litellm') when pricing lookup fails entirely, so callers
    can distinguish between a genuine $0.00 cost and an unknown cost.
    """
    # Strip all provider / region prefix segments so that
    # "bedrock/us-east-1/anthropic.claude-3-5-sonnet-20241022-v2:0"
    # → "anthropic.claude-3-5-sonnet-20241022-v2:0"
    # and "azure_ai/claude-sonnet-4-20250514" → "claude-sonnet-4-20250514".
    # Issue #658: model.split("/", 1)[-1] only strips one level; this strips
    # all leading non-model segments by taking the last "/"-separated token.
    bare_model = model.rsplit("/", 1)[-1] if "/" in model else model

    # Check overrides cache — try both full model string and bare name
    for key in (model, bare_model):
        if key in _cache:
            input_cost, output_cost = _cache[key]
            # Move to end for LRU tracking
            _cache.move_to_end(key)
            cost = float((input_cost * prompt_tokens + output_cost * completion_tokens) / Decimal(1_000_000))
            return cost, "override"

    # Fallback to litellm's built-in pricing
    if completion_response is not None:
        try:
            cost = litellm.completion_cost(completion_response=completion_response)
            return float(cost), "litellm"
        except LiteLLMNotFoundError:
            pass
        except Exception:
            log.error(
                "pricing.cost_calculation_error",
                model=model,
                source="completion_response",
                exc_info=True,
            )

    try:
        cost = litellm.completion_cost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(cost), "litellm"
    except LiteLLMNotFoundError:
        log.debug("pricing.model_not_supported", model=model)
        return None, "litellm"
    except Exception:
        log.error("pricing.cost_calculation_error", model=model, exc_info=True)
        return None, "litellm"


async def list_overrides(db: AsyncSession) -> PricingOverrideList:
    count_result = await db.execute(select(func.count(LLMPricingOverride.id)))
    total = count_result.scalar() or 0
    result = await db.execute(select(LLMPricingOverride).order_by(LLMPricingOverride.model_name))
    items = [PricingOverrideResponse.model_validate(row) for row in result.scalars().all()]
    return PricingOverrideList(items=items, total=total)


async def upsert_override(
    db: AsyncSession,
    data: PricingOverrideCreate,
    owner_id: uuid.UUID | None = None,
) -> tuple[PricingOverrideResponse, bool]:
    """Create or update a pricing override. Returns (response, created)."""
    result = await db.execute(select(LLMPricingOverride).where(LLMPricingOverride.model_name == data.model_name))
    existing = result.scalar_one_or_none()

    _warn_litellm_drift(data)

    if existing is not None:
        existing.input_cost_per_mtok = data.input_cost_per_mtok
        existing.output_cost_per_mtok = data.output_cost_per_mtok
        if data.effective_date is not None:
            existing.effective_date = data.effective_date
        existing.notes = data.notes
        await db.commit()
        await db.refresh(existing)
        await refresh_cache(db)
        return PricingOverrideResponse.model_validate(existing), False

    override = LLMPricingOverride(
        model_name=data.model_name,
        input_cost_per_mtok=data.input_cost_per_mtok,
        output_cost_per_mtok=data.output_cost_per_mtok,
        notes=data.notes,
        owner_id=owner_id,
    )
    if data.effective_date is not None:
        override.effective_date = data.effective_date
    db.add(override)
    await db.commit()
    await db.refresh(override)
    await refresh_cache(db)
    return PricingOverrideResponse.model_validate(override), True


async def delete_override(db: AsyncSession, model_name: str) -> bool:
    """Delete a pricing override. Returns True if it existed."""
    result = await db.execute(delete(LLMPricingOverride).where(LLMPricingOverride.model_name == model_name))
    await db.commit()
    if result.rowcount > 0:  # type: ignore[attr-defined]
        await refresh_cache(db)
        return True
    return False


def _warn_litellm_drift(data: PricingOverrideCreate) -> None:
    """Log a warning if override differs from litellm pricing by >50%."""
    try:
        litellm_input = litellm.completion_cost(model=data.model_name, prompt_tokens=1_000_000, completion_tokens=0)
        litellm_output = litellm.completion_cost(model=data.model_name, prompt_tokens=0, completion_tokens=1_000_000)
    except Exception:
        return

    override_input = float(data.input_cost_per_mtok)
    override_output = float(data.output_cost_per_mtok)

    for label, override_val, litellm_val in [
        ("input", override_input, litellm_input),
        ("output", override_output, litellm_output),
    ]:
        if litellm_val > 0:
            ratio = abs(override_val - litellm_val) / litellm_val
            if ratio > 0.5:
                log.warning(
                    "pricing.significant_drift",
                    model=data.model_name,
                    direction=label,
                    override=override_val,
                    litellm=litellm_val,
                    drift_pct=round(ratio * 100, 1),
                )
