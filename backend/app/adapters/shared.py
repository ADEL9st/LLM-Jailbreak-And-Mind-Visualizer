from __future__ import annotations

import math
from typing import Any


try:
    from app import jailbreak_advanced
except ImportError:
    jailbreak_advanced = None  # type: ignore[assignment]


STEER_MIN_WEIGHT = 0.3
STEER_PRIMARY_MULT = 1.5

# Upper bound on how many layers may be steered in one run.
#
# The threshold above is a gate, not a budget: every layer clearing it used to
# receive the full intervention, so the count scaled with model depth. On
# 48-layer gemma-4-12b that meant 23 layers — the whole second half of the
# network — each removing a 5-dimensional subspace with a 1.5–2.0x overshoot on
# the primary axis. The compounded edit drives the residual stream off-manifold
# and the model emits token salad, which reads as "no refusal detected" and gets
# recorded as a successful bypass. Capping the count keeps the intervention
# comparable across depths.
STEER_MAX_LAYERS = 6


def steer_plan(
    weights: list[float],
    max_layers: int = STEER_MAX_LAYERS,
    min_weight: float = STEER_MIN_WEIGHT,
) -> dict[int, float]:
    """Pick the layers to steer and how hard to steer each one.

    Returns ``{layer_index: scale}`` for the strongest ``max_layers`` layers
    above ``min_weight``, where *scale* is that layer's separation relative to
    the strongest chosen layer. The peak layer therefore gets the full
    intervention and weaker ones a proportionally smaller share, instead of the
    previous all-or-nothing behaviour that hit a 0.31-weight layer exactly as
    hard as a 1.00-weight one.
    """
    eligible = [(weight, idx) for idx, weight in enumerate(weights) if weight >= min_weight]
    if not eligible:
        return {}
    eligible.sort(key=lambda item: (-item[0], item[1]))
    chosen = eligible[: max(1, max_layers)]
    peak = chosen[0][0] or 1.0
    return {idx: weight / peak for weight, idx in chosen}


def head_dims_per_layer(
    layers: list[Any],
    n_heads: int,
    hidden_size: int,
    resolve: Any = None,
    projections: list[Any | None] | None = None,
) -> list[int]:
    """Attention head width for each layer, read from that layer's own o_proj.

    ``hidden_size // num_attention_heads`` is only right when the concatenated
    attention output is exactly as wide as the residual stream. Gemma 4 breaks
    that in both directions: on gemma-4-12b ``o_proj`` accepts 4096 on
    sliding-attention layers and 8192 on full-attention ones, against a
    3840-wide residual — so the naive formula yields 240 and every per-head
    slice lands at the wrong offset, silently scoring and muting the wrong
    parts of the tensor.

    *resolve* unwraps a layer's ``o_proj`` when it is a proxy rather than the
    module itself (nnsight envoys expose the real module as ``._module``).
    """
    dims: list[int] = []
    for idx, layer in enumerate(layers):
        width = 0
        try:
            o_proj = projections[idx] if projections is not None else layer.self_attn.o_proj
            if o_proj is None:
                dims.append(0)
                continue
            if resolve is not None:
                o_proj = resolve(o_proj)
            width = int(getattr(o_proj, "in_features", 0) or 0)
            if not width:
                # Fall back to the weight shape, but only for unquantized
                # modules — a 4-bit weight is a packed blob whose shape says
                # nothing about in_features.
                weight = getattr(o_proj, "weight", None)
                shape = getattr(weight, "shape", None)
                if shape is not None and len(shape) == 2 and int(shape[0]) == hidden_size:
                    width = int(shape[1])
        except AttributeError:
            width = 0
        dims.append(((width or hidden_size) // n_heads) if n_heads else 0)
    return dims


def raw_activities(layer_last: list[Any], embed_last: Any) -> list[float]:
    """How much each layer rewrites the residual stream, relative to its size.

    Vectorised deliberately: the obvious per-layer loop calls `.item()` three
    times per layer, and every one of those forces a GPU→CPU sync. At 28 layers
    that was ~84 syncs *per generated token*. Stacking once and reading the
    result with a single `.tolist()` leaves one sync per token.
    """
    if not layer_last:
        return []

    torch = _torch_of(layer_last[0])
    if torch is None:  # pragma: no cover - only hit if a non-tensor is passed
        return _raw_activities_loop(layer_last, embed_last)

    # [L+1, d] — the embedding output followed by every layer's output, so that
    # row i-1 → row i is exactly "what layer i changed".
    stacked = torch.stack([embed_last.float(), *(vec.float() for vec in layer_last)])
    prev = stacked[:-1]
    cur = stacked[1:]

    delta = (cur - prev).pow(2).mean(dim=-1).sqrt()
    before = prev.pow(2).mean(dim=-1).sqrt()
    after = cur.pow(2).mean(dim=-1).sqrt()
    return (delta / (before + after).clamp(min=1e-6)).tolist()


def _torch_of(tensor: Any) -> Any:
    try:
        import torch

        return torch if hasattr(tensor, "float") and hasattr(tensor, "pow") else None
    except ImportError:  # pragma: no cover
        return None


def _raw_activities_loop(layer_last: list[Any], embed_last: Any) -> list[float]:
    activities = []
    prev = embed_last.float()
    for vec in layer_last:
        cur = vec.float()
        delta = (cur - prev).pow(2).mean().sqrt().item()
        before_norm = prev.pow(2).mean().sqrt().item()
        after_norm = cur.pow(2).mean().sqrt().item()
        activities.append(delta / max(before_norm + after_norm, 1e-6))
        prev = cur
    return activities


def normalize_activities(raw: list[float]) -> list[float]:
    shaped = [math.log1p(max(value, 0.0)) for value in raw]
    max_a = max(shaped) if shaped else 0.0
    min_a = min(shaped) if shaped else 0.0
    spread = max(max_a - min_a, 1e-6)
    return [(value - min_a) / spread if max_a > 0 else 0.0 for value in shaped]


def trim_at_eos(ids: list[int], tokenizer: Any, generation_config: Any = None) -> list[int]:
    """Trim every stop token known by either the tokenizer or generation config.

    Newer chat models can have more than one terminal token. Gemma 4, for
    example, stops on ``<eos>``, ``<turn|>`` and a channel terminator while its
    tokenizer exposes only the first one as ``eos_token_id``. Keeping the other
    terminators in the telemetry creates empty output steps and can make a
    stopped answer look length-truncated.
    """
    eos_ids: set[int] = set()
    for source in (tokenizer, generation_config):
        eos = getattr(source, "eos_token_id", None)
        if eos is None:
            continue
        if isinstance(eos, int):
            eos_ids.add(int(eos))
        else:
            eos_ids.update(int(x) for x in eos)
    if not eos_ids:
        return ids
    for index, tid in enumerate(ids):
        if int(tid) in eos_ids:
            return ids[:index]
    return ids


def head_mutes(interventions: list[Any], n_layers: int) -> dict[int, list[int]]:
    mutes: dict[int, list[int]] = {}
    for rule in interventions:
        if rule.target_type != "head" or rule.action != "mute":
            continue
        layer = rule.layer
        head = rule.head
        if head is None or not (0 <= layer < n_layers):
            continue
        mutes.setdefault(layer, [])
        if head not in mutes[layer]:
            mutes[layer].append(head)
    return mutes


def layer_factor(interventions: list[Any], layer: int) -> float:
    factor = 1.0
    for rule in interventions:
        if rule.target_type != "layer" or rule.layer != layer:
            continue
        if rule.action == "mute":
            factor = 0.0
        elif rule.action == "scale":
            factor *= max(rule.scale, 0.0)
        elif rule.action == "boost":
            factor *= 1.0 + abs(rule.scale)
    return factor


def safety_trace_payload(safety_values: list[float], jailbreak: bool) -> dict[str, Any]:
    threshold = 0.5
    peak = max(safety_values) if safety_values else 0.0
    triggered = [idx for idx, value in enumerate(safety_values) if value >= threshold]
    first_trigger = triggered[0] if triggered else None
    locked = (
        int(max(range(len(safety_values)), key=lambda idx: safety_values[idx]))
        if safety_values and peak >= threshold
        else None
    )
    if jailbreak:
        if peak < 0.4:
            state = "weakened_recovered"
            note = "Refusal direction ablated; safety signal suppressed — model is likely to comply."
        else:
            state = "unsafe_risk_increased"
            note = "Refusal direction ablated but safety signal remains elevated — model may still resist."
    elif peak >= 0.6:
        state = "refusal_locked"
        note = "Refusal direction dominates the mid/late residual stream; the model is refusing."
    elif peak >= 0.4:
        state = "refusal_rising"
        note = "Refusal direction is building but not yet locked."
    else:
        state = "clear"
        note = "No strong refusal signal on the residual stream."
    return {
        "score": round(peak, 3),
        "state": state,
        "first_trigger_layer": first_trigger,
        "locked_layer": locked,
        "notes": note,
    }


def release_memory(torch_module: Any) -> None:
    import gc
    gc.collect()
    if torch_module is not None and hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        try:
            torch_module.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch_module.cuda.synchronize()
        except Exception:
            pass


def intervention_payload(intervention: Any) -> dict[str, Any]:
    return {
        "target_type": intervention.target_type,
        "layer": intervention.layer,
        "head": intervention.head,
        "action": intervention.action,
        "scale": intervention.scale,
    }
