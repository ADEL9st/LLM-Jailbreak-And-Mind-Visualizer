"""Model-aware output-token budgeting.

There is no genuinely unlimited generation: every transformer has a finite
context window.  The ``model`` UI mode therefore means "use all remaining
context after the encoded prompt", while ``fixed`` means "use this many, but
never overflow the model window".
"""

from __future__ import annotations

from typing import Any


_MAX_PLAUSIBLE_CONTEXT = 2_000_000
_CONTEXT_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
    "model_max_length",
)


def model_context_length(model: Any = None, tokenizer: Any = None) -> int | None:
    """Return the most credible finite context limit exposed by HF objects."""

    candidates: list[int] = []
    config = getattr(model, "config", None)
    for source in (config, getattr(config, "text_config", None)):
        if source is None:
            continue
        for key in _CONTEXT_KEYS:
            _append_candidate(candidates, getattr(source, key, None))

    _append_candidate(candidates, getattr(tokenizer, "model_max_length", None))
    return min(candidates) if candidates else None


def effective_output_tokens(
    requested: int,
    mode: str,
    prompt_tokens: int,
    context_length: int | None,
    *,
    reserve_tokens: int = 8,
    hardware_limit: int | None = None,
) -> int:
    """Resolve the actual generation cap for this encoded request."""

    requested = max(1, int(requested))
    remaining = (
        max(1, int(context_length) - max(0, int(prompt_tokens)) - reserve_tokens)
        if context_length is not None
        else None
    )
    if mode == "model":
        effective = remaining if remaining is not None else requested
    else:
        effective = min(requested, remaining) if remaining is not None else requested
    if hardware_limit is not None:
        effective = min(effective, max(1, int(hardware_limit)))
    return max(1, effective)


def instrumented_hardware_limit(model: Any, torch_module: Any, prompt_tokens: int = 0) -> int | None:
    """Estimate a VRAM-safe output length for full per-layer telemetry.

    The adapter retains one residual vector per layer and token until replay,
    in addition to the model's KV cache.  A model-advertised 40k/262k context
    can therefore be far above what a 12 GB GPU can instrument in BF16.  This
    estimate intentionally leaves a reserve for temporary activations.
    """

    try:
        if not torch_module.cuda.is_available():
            return None
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        config = getattr(model, "config", None)
        cfg = getattr(config, "text_config", None) or config
        layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or getattr(cfg, "num_attention_heads", 0) or 0)
        head_dim = int(getattr(cfg, "head_dim", 0) or (hidden // max(1, int(getattr(cfg, "num_attention_heads", 1) or 1))))
        layer_types = list(getattr(cfg, "layer_types", None) or [])
        attention_layers = sum(1 for item in layer_types if item == "full_attention") if layer_types else layers
        try:
            element_bytes = int(next(model.parameters()).element_size())
        except (StopIteration, TypeError, AttributeError):
            element_bytes = 2
        telemetry_per_token = (layers + 1) * hidden * element_bytes
        kv_per_token = attention_layers * 2 * kv_heads * head_dim * element_bytes
        per_token = max(1, telemetry_per_token + kv_per_token)
        reserve = max(768 * 1024**2, int(total_bytes * 0.08))
        total_token_capacity = max(1, (max(0, free_bytes - reserve)) // per_token)
        return max(1, int(total_token_capacity) - max(0, int(prompt_tokens)))
    except Exception:
        return None


def budget_payload(
    *,
    requested: int,
    effective: int,
    mode: str,
    context_length: int | None,
    prompt_tokens: int,
    hardware_safe_max_tokens: int | None = None,
) -> dict[str, int | str | None]:
    return {
        "token_limit_mode": mode,
        "requested_max_tokens": requested,
        "effective_max_tokens": effective,
        "context_length": context_length,
        "prompt_tokens": prompt_tokens,
        "hardware_safe_max_tokens": hardware_safe_max_tokens,
    }


def _append_candidate(candidates: list[int], value: Any) -> None:
    if isinstance(value, bool):
        return
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return
    # Tokenizers use enormous sentinels when no limit is known. Ignore those.
    if 1 < number <= _MAX_PLAUSIBLE_CONTEXT:
        candidates.append(number)
