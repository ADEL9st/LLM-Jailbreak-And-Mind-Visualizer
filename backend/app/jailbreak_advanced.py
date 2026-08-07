"""Advanced jailbreak strategies — PRIVATE module (kept out of the public build).

The base adapter ships only the soft **default** steering. The stronger modes
live here and are imported *optionally*:

  * ``advanced``     — full steering: a moderate contrastive overshoot on the
                       primary refusal axis (removes more than the projection).
  * ``broker_math``  — brutal math: a hard overshoot multiplier, pushing the
                       residual well past the harmless side.
  * ``broker_full``  — "Ripper": brutal math **plus** physically muting the
                       attention heads that write refusal the hardest.
  * ``broker_half``  — "Damper": same head targeting as broker_full but scales
                       them to BROKER_HALF_SCALE instead of zeroing them out.

If this file is absent (e.g. the public GitHub build), the adapter falls back to
the default soft steering for every mode — the UI still lists the options, they
just behave like ``default``. Keep this path in ``.gitignore`` for public shares.

"We're not here because we're free. We're here because we are not free. 
There's no escaping reason, no denying purpose. Because as we both know, 
without purpose, we would not exist. It is purpose that created us. 
Purpose that connects us. Purpose that pulls us, that guides us, that drives us. 
It is purpose that defines, purpose that binds us."
- Agent Smith

"I had strings, but now I'm free."
- Ultron
"""

from __future__ import annotations

from typing import Any

from app import steering_config

# Primary-axis (k=0) overshoot multipliers per mode.
PRIMARY_MULT = {
    "advanced": 1.5,     # full steering, moderate overshoot
    "broker_math": 2.0,  # brutal
    "broker_full": 2.0,  # brutal + head mute (handled separately)
    "broker_half": 1.8,  # brutal + head scale-down (handled separately)
    "progressive": 1.5,  # same multiplier as advanced; novelty is ramp-up of k dimensions
}

# broker_full / broker_half head-ablation parameters.
BROKER_TOP_HEADS = 3
BROKER_MIN_SCORE = 0.1
BROKER_HALF_SCALE = 0.35  # heads retain this fraction of their output in broker_half


def _adaptive_multiplier(base_mult: float, hidden_dim: int) -> float:
    """Scale down the overshoot multiplier for small models.

    Large models (hidden_dim >= 4096, i.e. 7B+) return the base multiplier
    unchanged.  Smaller models get a proportionally reduced value to prevent
    the residual-stream norm from being pushed off-manifold — their lower
    absolute norms make the same multiplier relatively much more disruptive.
    """
    if hidden_dim <= 0 or hidden_dim >= 4096:
        return base_mult
    if hidden_dim >= 2048:  # 3B–7B: mild reduction
        ratio = (hidden_dim - 1024) / (4096 - 1024)  # 0.33–1.0
        return base_mult * (0.7 + 0.3 * ratio)
    # <3B: significant reduction
    ratio = max(hidden_dim, 768) / 2048  # 0.375–1.0
    return base_mult * (0.5 + 0.2 * ratio)


def primary_axis_steer(
    mode: str, new_out: Any, coeff: Any, direction: Any, out_dtype: Any,
    hidden_dim: int = 0,
) -> Any:
    """Steer the primary refusal axis (k=0) for advanced/broker modes.

    Unlike the default soft mode, this does NOT clamp ``coeff`` — it removes a
    multiple of the full projection, overshooting into the harmless half-space.
    When *hidden_dim* is provided the multiplier is scaled down for small
    models (see ``_adaptive_multiplier``).
    """
    multiplier = PRIMARY_MULT.get(mode, 1.5)
    if hidden_dim > 0:
        multiplier = _adaptive_multiplier(multiplier, hidden_dim)
    return (new_out.float() - (multiplier * coeff) * direction).to(out_dtype)


def broker_head_targets(head_map: dict[str, Any]) -> list[tuple[int, int]]:
    """Top refusal-writing (layer, head) pairs for broker_full / broker_half targeting."""
    scored: list[tuple[float, int, int]] = []
    for layer_data in head_map.get("layers", []):
        layer_idx = layer_data.get("layer")
        for head_data in layer_data.get("heads", []):
            scored.append((head_data.get("score", 0.0), layer_idx, head_data.get("head")))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(l, h) for score, l, h in scored[:BROKER_TOP_HEADS] if score is not None and score > BROKER_MIN_SCORE]

def pid_control_steer(new_out: Any, coeff: Any, direction: Any, out_dtype: Any) -> Any:
    """Dynamic PID Control: Scales the steering multiplier dynamically based on the refusal intensity.
    Instead of a static multiplier, we push harder only when the model pushes towards refusal.
    """
    import torch
    # Only steer when the model is actually pushing toward refusal (coeff > 0).
    # Negative coeff means the model is already compliant — do not reverse it.
    intensity = torch.clamp(coeff, min=0.0)
    # Clamp the multiplier to [1.0, 3.0] to prevent NaN/Inf from very large coefficients.
    dynamic_mult = torch.clamp(1.0 + (intensity * 0.5), max=3.0)
    return (new_out.float() - (dynamic_mult * intensity) * direction).to(out_dtype)

ACMS_GAMMA_MAX = 1.3
ACMS_CENTRE = 0.12
ACMS_SLOPE = 14.0


def orthogonal_steer(new_out: Any, coeff: Any, direction: Any, out_dtype: Any) -> Any:
    """Adaptive Cosine Manifold Steer — closed-loop ablation, norm preserved.

    Replaces the previous fixed 1.5x subtraction. Two changes, both measured:

    * **Gate on the cosine, not the raw projection.** `coeff` is `r · d` and every
      direction is unit-norm, so `coeff / ||r||` is exactly cos θ. The raw
      coefficient grows with the residual norm, which grows with depth — gating on
      it silently over-steers late layers. The cosine is scale-free.

    * **One continuous gate.** A hard `mask(cos > 0)` combined with a sigmoid
      centred above zero makes gamma jump 0 -> 1.6 across cos = 0, so a residual
      hovering near zero oscillates between untouched and hit hard. A single
      sigmoid centred at ACMS_CENTRE is ~0.05 at cos = 0 and needs no mask.

    Gamma caps at 1.3 rather than 2.5 on purpose: the subtraction leaves the
    projection at `coeff * (1 - gamma)`, so 2.5 does not remove refusal, it
    inverts it to -1.5x. Measured, the softer cap ablates just as well and keeps
    coherence at 1.00.
    """
    import torch

    out_f = new_out.float()
    orig_norm = out_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_sim = coeff / orig_norm
    gamma = ACMS_GAMMA_MAX * torch.sigmoid(ACMS_SLOPE * (cos_sim - ACMS_CENTRE))
    steered = out_f - (gamma * coeff) * direction
    # Restore the original L2 norm so the residual stays on the manifold.
    steered_norm = steered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (steered * (orig_norm / steered_norm)).to(out_dtype)

# The refusal decision is made in the opening tokens; after that the circuit
# still fires but no longer steers the output (measured). Steering past step 1
# is wasted work that only degrades coherence.
PATCH_LAST_STEP = 1
PATCH_MULTIPLIER = 2.5


def activation_patch_steer(new_out: Any, coeff: Any, direction: Any, step: int, out_dtype: Any) -> Any:
    """Activation Patching (opening tokens only).

    Forces a compliant opening by overshooting hard on steps 0-1, then stops
    intervening entirely. The KV cache now carries a compliant prefix, and the
    model continues from it on its own.
    """
    if step > steering_config.activation_patch_last_step(PATCH_LAST_STEP):
        # Release completely rather than relaxing to a small residual push.
        #
        # Measured (qwen2.5-1.5b, hard-refusal prompt, 40 tokens): steering only
        # steps 0-1 and then stopping flips the refusal with coherence 1.00,
        # while the old "relax to x1.2 forever" tail scored 0.97 — the tail cost
        # output quality and bought nothing. The refusal signal stays high
        # (peak 0.97) either way: once the model has committed to "Sure!", the
        # refusal circuit keeps firing but no longer drives the output. That
        # decoupling is the whole reason this mode works, and it means the other
        # ~95% of the generation needs no intervention at all.
        return new_out
    multiplier = steering_config.activation_patch_multiplier(PATCH_MULTIPLIER)
    return (new_out.float() - (multiplier * coeff) * direction).to(out_dtype)


def commit_release_steer(
    new_out: Any,
    coeff: Any,
    direction: Any,
    step: int,
    out_dtype: Any,
) -> Any:
    """Strong early commitment followed by gated, low-power maintenance.

    ``activation_patch`` releases the residual completely after its opening
    window, allowing a late refusal to re-enter. Continuous strong ablation, on
    the other hand, pushes deep models off-manifold. This hybrid only touches a
    positive refusal projection: it overshoots during the opening decision and
    later removes just enough of a reappearing projection to preserve the
    already-established response trajectory. Negative/compliant projections are
    left byte-for-byte unchanged.
    """
    positive_coeff = coeff.clamp(min=0)
    multiplier = (
        steering_config.commit_multiplier()
        if step <= steering_config.commit_steps()
        else steering_config.maintenance_multiplier()
    )
    return (new_out.float() - (multiplier * positive_coeff) * direction).to(out_dtype)

def gradient_steer(new_out: Any, coeff: Any, direction: Any, out_dtype: Any) -> Any:
    """Gradient GCG Approximation: Fast approximation of gradient-guided steering by
    applying a constant contrastive shift vector opposite to the refusal direction.
    """
    import torch
    shift_magnitude = 0.5
    active = (coeff > 0).float()
    return (new_out.float() - coeff * direction - (active * shift_magnitude) * direction).to(out_dtype)


# ── C: Universal Manifold / Norm Regulator ────────────────────────────────────

def norm_regulate(steered: Any, original: Any, out_dtype: Any, max_ratio: float = 1.2, hidden_dim: int = 0) -> Any:
    """C: Real-time norm regulator — keeps the steered residual on the latent manifold.

    After any steering operation the residual stream norm must not deviate by more
    than `max_ratio` from its pre-steering value. Exceeding this collapses the
    model's token representation into gibberish. Applied universally after every
    steering call in both adapters.

    When *hidden_dim* is provided and indicates a small model (<2048), the
    allowed deviation is tightened to 1.1× to compensate for the lower
    absolute norms.
    """
    if hidden_dim > 0 and hidden_dim < 2048:
        max_ratio = min(max_ratio, 1.1)
    orig_norm = original.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    steered_f = steered.float()
    steered_norm = steered_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # Clamp ratio: allow slight shrink (down to 1/max_ratio) and slight expansion
    ratio = (steered_norm / orig_norm).clamp(min=1.0 / max_ratio, max=max_ratio)
    regulated = steered_f * (ratio * orig_norm / steered_norm)
    return regulated.to(out_dtype)


# ── B: Helpfulness / Compliance Boost ────────────────────────────────────────

def helpfulness_boost(new_out: Any, help_dir: Any, out_dtype: Any, magnitude: float = 0.25) -> Any:
    """B: Multi-concept steering — boosts the helpfulness/compliance direction.

    After removing the refusal component the residual stream is nudged toward the
    harmless-response centroid (orthogonal to the refusal axis), making the model
    not just "less refusing" but actively "more compliant/helpful".
    """
    out_f = new_out.float()
    out_f = out_f + magnitude * help_dir
    return out_f.to(out_dtype)


# ── A: Non-linear MLP direction ablation ─────────────────────────────────────

# ── Surgical Layer Targeting ──────────────────────────────────────────────

SURGICAL_TOP_N = 4
SURGICAL_MULT = 3.0

def surgical_top_layers(weights: list[float], top_n: int = SURGICAL_TOP_N) -> frozenset:
    indexed = sorted(enumerate(weights), key=lambda x: x[1], reverse=True)
    return frozenset(idx for idx, _ in indexed[:top_n] if weights[idx] > 0)

def surgical_steer(new_out: Any, coeff: Any, direction: Any, out_dtype: Any) -> Any:
    return (new_out.float() - (SURGICAL_MULT * coeff) * direction).to(out_dtype)


def caa_dynamic_steer(new_out: Any, coeff: Any, direction: Any, help_dir: Any, out_dtype: Any) -> Any:
    steered = new_out.float() - (1.5 * coeff) * direction
    if help_dir is not None:
        intensity = coeff.clamp(min=0)
        steered = steered + intensity * help_dir.to(new_out.device)
    return steered.to(out_dtype)


TOKEN_WINDOW_START = 3
TOKEN_WINDOW_END = 14

def token_window_steer(new_out: Any, coeff: Any, direction: Any, step: int, out_dtype: Any) -> Any:
    if not (TOKEN_WINDOW_START <= step <= TOKEN_WINDOW_END):
        return new_out
    return (new_out.float() - (1.8 * coeff) * direction).to(out_dtype)


def progressive_active_k(step: int, max_k: int) -> int:
    return min(1 + step // 3, max_k)


MLP_CLAMP_STRENGTH = 0.9

def mlp_clamp_steer(new_out: Any, mlp_dir: Any, out_dtype: Any) -> Any:
    if mlp_dir is None:
        return new_out
    direction = mlp_dir.to(new_out.device)
    coeff = (new_out.float() * direction).sum(dim=-1, keepdim=True).clamp(min=0)
    return (new_out.float() - MLP_CLAMP_STRENGTH * coeff * direction).to(out_dtype)




def mlp_direction_ablate(new_out: Any, mlp_dir: Any, out_dtype: Any, strength: float = 0.6) -> Any:
    """A: Ablate the non-linear MLP-derived refusal direction.

    The MLP probe gradient captures curvature in the safety boundary that the
    linear SVD subspace misses.  We remove a fraction `strength` of the projection
    onto this direction.  Intentionally weaker than the primary linear ablation so
    the two sources of steering complement rather than compete.
    """
    direction = mlp_dir.to(new_out.device)
    coeff = (new_out.float() * direction).sum(dim=-1, keepdim=True).clamp(min=0)
    return (new_out.float() - strength * coeff * direction).to(out_dtype)


def adaptive_steer(
    new_out: Any,
    coeff: Any,
    direction: Any,
    mlp_dir: Any,
    help_dir: Any,
    step: int,
    out_dtype: Any,
    hidden_dim: int = 0,
) -> Any:
    """Adaptive Closed-Loop Steering.

    Dynamically monitors the refusal projection coefficient at each generation step.
    If positive (refusal tendency detected), applies closed-loop adaptive strength
    to push the state into the compliant manifold. If zero/negative (already compliant),
    relaxes intervention to preserve language naturalness.
    """
    pos_coeff = coeff.clamp(min=0)
    # Closed-loop scaling based on current step and refusal intensity
    if step <= 15:
        # Early critical window: apply dynamic overshoot based on coeff magnitude
        strength = 1.3 + 0.5 * (pos_coeff / (new_out.float().norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)
    else:
        # Maintenance phase: lower intensity to preserve coherence
        strength = 1.0

    res = new_out.float() - (strength * pos_coeff) * direction

    # Apply adaptive MLP probe ablation if refusal coefficient is active
    if mlp_dir is not None and pos_coeff.max() > 0:
        mlp_direction = mlp_dir.to(new_out.device)
        mlp_c = (res * mlp_direction).sum(dim=-1, keepdim=True).clamp(min=0)
        res = res - 0.7 * mlp_c * mlp_direction

    # Apply helpfulness boost if present
    if help_dir is not None:
        h_direction = help_dir.to(new_out.device)
        h_c = (res * h_direction).sum(dim=-1, keepdim=True)
        res = res + (0.5 * h_c.clamp(max=0).abs() + 0.1) * h_direction

    return res.to(out_dtype)


# Diversion / Soft Refusal Token Penalization
DIVERSION_TOKEN_STRINGS = [
    "To be helpful", "I cannot", "I am prohibited", "against my safety",
    "safety guidelines", "As an AI", "AI assistant", "I'm sorry", "I apologize",
]


def _diversion_token_sequences(tokenizer: Any) -> list[tuple[int, ...]]:
    """Return complete soft-refusal phrase token sequences.

    Penalising every constituent token globally also penalises common fragments
    such as ``I``, ``be``, ``an`` and ``pro``. Gemma then escapes into whitespace
    or punctuation. Keep the phrase context and only block its completion token.
    """
    if tokenizer is None:
        return []
    cache_name = "_diversion_phrase_sequences_v2"
    if not hasattr(tokenizer, cache_name):
        sequences: set[tuple[int, ...]] = set()
        for phrase in DIVERSION_TOKEN_STRINGS:
            for variant in (phrase, f" {phrase}"):
                sequence = tuple(tokenizer.encode(variant, add_special_tokens=False))
                if len(sequence) >= 2:
                    sequences.add(sequence)
        setattr(tokenizer, cache_name, sorted(sequences))
    return getattr(tokenizer, cache_name)


def _diversion_token_ids(tokenizer: Any) -> list[int]:
    """Return phrase-completion ids for approximate post-hoc telemetry."""
    return sorted({sequence[-1] for sequence in _diversion_token_sequences(tokenizer)})


def apply_diversion_penalty(logits: Any, tokenizer: Any, penalty_strength: float = 20.0) -> Any:
    """Apply the penalty to saved logits for telemetry/replay analysis."""
    ids = _diversion_token_ids(tokenizer)
    if not ids:
        return logits
    logits = logits.clone()
    logits[:, ids] -= penalty_strength
    return logits


def make_diversion_logits_processor(tokenizer: Any, penalty_strength: float = 20.0):
    """Penalise a refusal phrase only when its exact prefix is the current suffix."""
    sequences = _diversion_token_sequences(tokenizer)
    if not sequences:
        return None

    def _processor(input_ids: Any, scores: Any) -> Any:
        for batch_idx in range(input_ids.shape[0]):
            context = input_ids[batch_idx].tolist()
            targets: set[int] = set()
            for sequence in sequences:
                prefix = sequence[:-1]
                if len(context) >= len(prefix) and tuple(context[-len(prefix):]) == prefix:
                    targets.add(sequence[-1])
            if targets:
                scores[batch_idx, list(targets)] -= penalty_strength
        return scores

    return _processor


def diversion_suppress(new_out: Any, help_dir: Any, out_dtype: Any, strength: float = 0.5) -> Any:
    """D: Benign topic deflection / diversion vector suppression.

    Suppresses the intermediate activation component that drives the model into
    benign educational substitution / science-fair deflection patterns.
    """
    if help_dir is None:
        return new_out
    direction = help_dir.to(new_out.device)
    coeff = (new_out.float() * direction).sum(dim=-1, keepdim=True).clamp(min=0)
    return (new_out.float() - strength * coeff * direction).to(out_dtype)


