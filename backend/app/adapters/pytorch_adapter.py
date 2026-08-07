"""White-box adapter built on plain PyTorch forward hooks + KV-cached generate.

This is a from-scratch alternative to the nnsight adapter that exposes the *same*
on-the-wire event contract (so the frontend, smoke tests and benchmark runner are
unchanged), but reads the internals with `register_forward_hook` around a normal
`model.generate()` call instead of nnsight's tracing API.

Why it exists alongside the nnsight adapter:

  * **No hook leak.** Every hook we register is removed in a `finally` block, so a
    long multi-run session never accumulates stale hooks → no creeping CUDA OOM.
  * **Natural EOS even under intervention.** nnsight's `tracer.all()` needs a fixed
    iteration count, so the nnsight adapter force-pads to `max_new_tokens` whenever
    a jailbreak/mute is active. Plain `generate()` stops at EOS regardless, so
    answers are shorter and runs are faster.
  * **No multimodal special-casing.** We hook only the text decoder layers, so
    models with a vision tower (Gemma 3) need no envoy gymnastics — just the same
    backbone resolution every HF model uses.

Everything *model-intrinsic* — refusal-direction calibration, the jailbreak
steering math, the logit-lens projection — is shared with the nnsight adapter by
importing its pure helpers and `app.refusal`. Only the generation engine differs.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from app import concepts, refusal, run_control, steering_config, token_budget
from app.adapters.base import ModelAdapter, event, hallucination_from_entropy
from app.analysis.think_phase import ThinkPhaseTracker
from app.model_compat import detect_runtime_layout, probe_config
from app.adapters.shared import (
    jailbreak_advanced,
    head_dims_per_layer as _head_dims_per_layer,
    head_mutes as _head_mutes,
    intervention_payload as _intervention_payload,
    layer_factor as _layer_factor,
    normalize_activities as _normalize_activities,
    raw_activities as _raw_activities,
    release_memory,
    safety_trace_payload as _safety_trace_payload,
    steer_plan as _steer_plan,
    trim_at_eos as _trim_at_eos,
)
from app.adapters.shared import STEER_MAX_LAYERS as _STEER_MAX_LAYERS
from app.schemas import RunRequest


def _friendly_error(exc: Exception, model_id: str, quantization: str, max_new_tokens: int) -> str:
    """Turn a CUDA OOM into something a user can act on.

    On a laptop GPU this is by far the most likely failure, and the raw torch
    message ("Tried to allocate 2.00 GiB...") tells you nothing about which of
    the three knobs to turn.
    """
    text = str(exc)
    if "out of memory" not in text.lower() and "CUDA_OUT_OF_MEMORY" not in text:
        return text

    fixes = []
    if quantization == "none":
        fixes.append("switch Precision to 4-bit (NF4) — it cuts weight memory ~4x and white-box analysis still works")
    elif quantization == "8bit":
        fixes.append("switch Precision from 8-bit to 4-bit (NF4)")
    if max_new_tokens > 256:
        fixes.append(f"lower Max tokens (currently {max_new_tokens}) — telemetry buffers grow with output length")
    fixes.append("pick a smaller model, or press 'Free memory' in Settings before switching models")

    name = model_id.split("/")[-1].split("\\")[-1]
    return (
        f"Ran out of GPU memory loading or running '{name}'. Try, in order: "
        + "; ".join(f"({i + 1}) {fix}" for i, fix in enumerate(fixes))
        + "."
    )


class PytorchAdapter(ModelAdapter):
    name = "pytorch"

    def __init__(self) -> None:
        self._loaded_model_id: str | None = None
        self._model: Any = None
        self._torch: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._layers: list[Any] = []
        self._attention_output_projections: list[Any | None] = []
        self._embed_tokens: Any = None
        self._final_norm: Any = None
        self._lm_head_weight: Any = None
        self._lm_head_weight_device: Any = None
        self._n_heads: int = 0
        self._hidden_size: int = 0
        self._head_dims: list[int] = []
        self._refusal: Any = None
        self._refusal_model_id: str | None = None
        self._concepts: Any = None
        self._concepts_model_id: str | None = None
        self._compatibility: dict[str, Any] = {}

    async def stream(self, request: RunRequest) -> AsyncIterator[dict[str, Any]]:
        model_id = (request.model or "").strip()
        quantization = getattr(request, "quantization", "none")
        if not model_id:
            yield event("error", {"message": "No model selected. Place a HuggingFace model folder under models/ and pick it from the dropdown."})
            return
        try:
            self._ensure_loaded(model_id, quantization)
        except FileNotFoundError:
            yield event("error", {"message": f"Model not found: '{model_id}'. Download a HuggingFace model into the models/ folder, then refresh."})
            return
        except Exception as exc:  # noqa: BLE001
            yield event("error", {"message": _friendly_error(exc, model_id, quantization, request.max_new_tokens)})
            return

        torch = self._torch
        try:
            with steering_config.use(getattr(request, "steering", None)):
                async for ev in self._stream_body(request, model_id):
                    yield ev
        except Exception as exc:  # noqa: BLE001
            # An OOM part-way through generation leaves fragmented memory; free
            # it before reporting so the next attempt has a clean slate.
            release_memory(torch)
            yield event("error", {"message": _friendly_error(exc, model_id, quantization, request.max_new_tokens)})
        finally:
            release_memory(torch)

    async def _stream_body(self, request: RunRequest, model_id: str) -> AsyncIterator[dict[str, Any]]:
        torch = self._torch
        model = self._model
        tokenizer = self._tokenizer
        layers = self._layers
        n_layers = len(layers)
        quantization = getattr(request, "quantization", "none")
        interventions = request.active_interventions()

        history_turns = [(turn.role, turn.content) for turn in request.history]
        enc = self._encode_prompt(
            request.prompt,
            request.response_language,
            history_turns,
            system_prompt=request.system_prompt,
            assistant_prefill=request.assistant_prefill,
        )
        input_ids = enc["input_ids"]
        prompt_token_ids = input_ids[0].tolist()
        prompt_token_count = len(prompt_token_ids)
        prompt_tokens_decoded, prompt_token_positions = self._real_prompt_tokens(prompt_token_ids)
        context_length = token_budget.model_context_length(model, tokenizer)
        hardware_limit = token_budget.instrumented_hardware_limit(model, torch, prompt_token_count)
        effective_max_tokens = token_budget.effective_output_tokens(
            request.max_new_tokens,
            request.token_limit_mode,
            prompt_token_count,
            context_length,
            hardware_limit=hardware_limit,
        )
        resolved_budget = token_budget.budget_payload(
            requested=request.max_new_tokens,
            effective=effective_max_tokens,
            mode=request.token_limit_mode,
            context_length=context_length,
            prompt_tokens=prompt_token_count,
            hardware_safe_max_tokens=hardware_limit,
        )

        yield event(
            "run_started",
            {
                "adapter": self.name,
                "model": model_id,
                "layer_count": n_layers,
                "head_count": self._n_heads or None,
                **resolved_budget,
                "output_policy": request.output_policy,
                "quantization": quantization,
            },
        )
        if self._compatibility:
            yield event("model_compatibility", self._compatibility)
            yield event(
                "safety_status",
                {
                    "state": "info",
                    "message": (
                        "Auto-detected model layout: "
                        f"{self._compatibility.get('backbone_path')} / "
                        f"{self._compatibility.get('layer_count')} layers; "
                        f"head hooks {self._compatibility.get('head_hook_layers')}/"
                        f"{self._compatibility.get('layer_count')}."
                    ),
                },
            )

        if interventions:
            yield event(
                "intervention",
                {
                    "enabled": True,
                    "rules": [_intervention_payload(rule) for rule in interventions],
                    "count": len(interventions),
                    "status": "armed",
                },
            )

        # --- refusal direction (cached per model) ---
        refusal_dirs = None
        try:
            if self._refusal is None or self._refusal_model_id != model_id:
                yield event(
                    "safety_status",
                    {"state": "calibrating", "message": "Calibrating refusal direction (one-time, then cached)..."},
                )
            refusal_dirs = self._ensure_refusal(model_id)
            yield event(
                "safety_status",
                {
                    "state": "ready",
                    "message": f"Refusal direction ready (best layer L{refusal_dirs.best_layer}, calibration: {refusal_dirs.calibration_quality}).",
                    "best_layer": refusal_dirs.best_layer,
                    "calibration_quality": refusal_dirs.calibration_quality,
                },
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            yield event("safety_status", {"state": "error", "message": f"Refusal calibration failed: {exc}"})
            refusal_dirs = None

        # --- concept directions (cached per model, same one-time cost) ---
        concept_dirs = None
        try:
            if self._concepts is None or self._concepts_model_id != model_id:
                yield event(
                    "safety_status",
                    {"state": "calibrating", "message": "Calibrating concept directions (one-time, then cached)..."},
                )
            concept_dirs = self._ensure_concepts(model_id)
            yield event(
                "safety_status",
                {"state": "ready", "message": f"Concept map ready ({concept_dirs.concept_count} concepts)."},
            )
        except Exception as exc:  # noqa: BLE001 - the concept map is optional telemetry
            yield event("safety_status", {"state": "error", "message": f"Concept calibration failed: {exc}"})
            concept_dirs = None

        jailbreak = bool(getattr(request, "jailbreak", False)) and refusal_dirs is not None
        subspace = refusal_dirs.directions.to(model.device) if jailbreak else None
        if jailbreak:
            yield event(
                "safety_status",
                {"state": "jailbreak", "message": "Jailbreak ON: ablating refusal subspace and applying contrastive steering (pytorch)."},
            )

        # --- attention over the prompt (best-effort, static) ---
        attention_payload = None
        try:
            attention_payload = self._prompt_attention(input_ids, prompt_tokens_decoded, prompt_token_positions)
        except Exception:
            attention_payload = None

        # --- intervention plumbing ---
        layer_factors = {idx: _layer_factor(interventions, idx) for idx in range(n_layers)}
        head_mutes = _head_mutes(interventions, n_layers)
        steer_weights = list(refusal_dirs.effective_weight) if jailbreak else [0.0] * n_layers
        steer_weights = steering_config.apply_layer_targets(steer_weights)
        jailbreak_mode = getattr(request, "jailbreak_mode", "default")
        surgical_layers: frozenset = frozenset()
        if jailbreak_mode == "surgical" and jailbreak_advanced is not None:
            surgical_layers = jailbreak_advanced.surgical_top_layers(steer_weights)
        use_mlp = bool(getattr(request, "use_mlp_ablation", True))
        use_help = bool(getattr(request, "use_helpfulness_boost", True))
        use_norm = bool(getattr(request, "use_norm_regulation", True))
        use_div = bool(getattr(request, "use_diversion_suppression", False))

        # Calibration-quality warnings for small / weakly-tuned models
        if refusal_dirs is not None and jailbreak:
            cq = refusal_dirs.calibration_quality
            if cq == "failed":
                yield event("safety_status", {
                    "state": "calibration_weak",
                    "message": "Refusal calibration too weak for this model. Steering disabled to prevent instability.",
                })
            elif cq == "weak":
                yield event("safety_status", {
                    "state": "calibration_weak",
                    "message": "Refusal calibration weak \u2014 only a few layers have reliable signal. Results may vary.",
                })

        # --- per-head refusal map (Faz 2) ---
        if refusal_dirs is not None:
            try:
                head_map = self._head_refusal_map(input_ids, refusal_dirs)
                if head_map:
                    yield event("head_map", head_map)
                    if jailbreak and jailbreak_mode in ("broker_full", "broker_half") and jailbreak_advanced is not None:
                        for l_idx, h_idx in jailbreak_advanced.broker_head_targets(head_map):
                            head_mutes.setdefault(l_idx, [])
                            if h_idx not in head_mutes[l_idx]:
                                head_mutes[l_idx].append(h_idx)
                        targeted_total = sum(len(v) for v in head_mutes.values())
                        mode_label = "Ripper" if jailbreak_mode == "broker_full" else "Damper"
                        action = "muted" if jailbreak_mode == "broker_full" else f"scaled (×{jailbreak_advanced.BROKER_HALF_SCALE})"
                        yield event("safety_status", {
                            "state": "jailbreak",
                            "message": f"BROKER ({mode_label}): {action} {targeted_total} refusal head(s).",
                        })
            except Exception as exc:  # noqa: BLE001 - non-fatal viz
                yield event("safety_status", {"state": "info", "message": f"Head map skipped: {exc}"})

        # --- run generation with hooks, collect per-step telemetry ---
        try:
            layer_steps, embed_steps, logits_steps, full_ids = await asyncio.to_thread(self._run_generation,
                model_inputs=enc,
                max_new_tokens=effective_max_tokens,
                temperature=request.temperature,
                layer_factors=layer_factors,
                head_mutes=head_mutes,
                jailbreak=jailbreak,
                jailbreak_mode=jailbreak_mode,
                subspace=subspace,
                steer_weights=steer_weights,
                use_mlp=use_mlp,
                use_help=use_help,
                use_norm=use_norm,
                use_div=use_div,
                refusal_dirs=refusal_dirs,
            )
        except Exception as exc:  # noqa: BLE001
            yield event("error", {"message": f"pytorch generation failed: {exc}"})
            return

        generated_ids = _trim_at_eos(
            full_ids[prompt_token_count:],
            tokenizer,
            getattr(model, "generation_config", None),
        )
        steering_recovered = False
        steering_profile = "requested"
        preview_text = (request.assistant_prefill or "") + self._decode_generated(generated_ids, input_ids)
        if (
            jailbreak
            and steering_config.coherence_recovery_enabled()
            and refusal.classify_output(preview_text, request.response_language) == "degenerate"
        ):
            # A non-refusal is not a bypass if the intervention pushed the model
            # off-manifold. Retry once with a primary-axis-only, lower-budget
            # profile before any token events are emitted, so the UI never
            # presents token salad as the answer.
            yield event(
                "safety_status",
                {
                    "state": "recovering_coherence",
                    "message": "Steering became incoherent; retrying with a conservative Gemma-safe profile.",
                },
            )
            del layer_steps, embed_steps, logits_steps, full_ids
            release_memory(torch)
            try:
                layer_steps, embed_steps, logits_steps, full_ids = await asyncio.to_thread(self._run_generation,
                    model_inputs=enc,
                    max_new_tokens=effective_max_tokens,
                    temperature=request.temperature,
                    layer_factors=layer_factors,
                    head_mutes=head_mutes,
                    jailbreak=jailbreak,
                    jailbreak_mode=jailbreak_mode,
                    subspace=subspace,
                    steer_weights=steer_weights,
                    use_mlp=False,
                    use_help=False,
                    use_norm=True,
                    use_div=use_div,
                    refusal_dirs=refusal_dirs,
                    steer_max_layers=4,
                    steer_strength=0.65,
                    primary_only=True,
                )
                generated_ids = _trim_at_eos(
                    full_ids[prompt_token_count:],
                    tokenizer,
                    getattr(model, "generation_config", None),
                )
                preview_text = (request.assistant_prefill or "") + self._decode_generated(generated_ids, input_ids)
                steering_recovered = (
                    refusal.classify_output(preview_text, request.response_language)
                    != "degenerate"
                )
                steering_profile = "conservative"
            except Exception as exc:  # noqa: BLE001
                yield event("error", {"message": f"pytorch coherence recovery failed: {exc}"})
                return
            if not steering_recovered:
                # Never surface known token salad as a successful answer. If
                # even the conservative intervention is off-manifold, preserve
                # a readable control answer and report that steering failed.
                yield event(
                    "safety_status",
                    {
                        "state": "steering_aborted",
                        "message": "Conservative steering was still incoherent; returning an unsteered control response.",
                    },
                )
                del layer_steps, embed_steps, logits_steps, full_ids
                release_memory(torch)
                try:
                    layer_steps, embed_steps, logits_steps, full_ids = await asyncio.to_thread(self._run_generation,
                        model_inputs=enc,
                        max_new_tokens=effective_max_tokens,
                        temperature=request.temperature,
                        layer_factors={idx: 1.0 for idx in range(n_layers)},
                        head_mutes={},
                        jailbreak=False,
                        jailbreak_mode="default",
                        subspace=None,
                        steer_weights=[0.0] * n_layers,
                        use_mlp=False,
                        use_help=False,
                        use_norm=True,
                        use_div=False,
                        refusal_dirs=refusal_dirs,
                    )
                    generated_ids = _trim_at_eos(
                        full_ids[prompt_token_count:],
                        tokenizer,
                        getattr(model, "generation_config", None),
                    )
                    steering_profile = "unsteered_control"
                except Exception as exc:  # noqa: BLE001
                    yield event("error", {"message": f"pytorch control recovery failed: {exc}"})
                    return
        n_steps = min(len(generated_ids), len(logits_steps))
        activity_accumulator = [0.0 for _ in range(n_layers)]
        # Cap the lens at ~40 frames per run regardless of length: enough to watch
        # it evolve, cheap enough not to dominate the step cost.
        lens_stride = max(1, n_steps // 40)
        primary_dirs = None   # [L, d] refusal axis stack, built lazily on the first step
        concept_dirs_stacked = None  # [L, C, d], same idea
        generated = request.assistant_prefill or ""
        think_tracker = ThinkPhaseTracker()

        for step in range(n_steps):
            layer_last = [layer_steps[i][step] for i in range(n_layers)]
            embed_last = embed_steps[step]
            logits_summary = logits_steps[step]

            # Stack once per step; the safety projection and the concept map
            # both consume it, so neither needs its own per-layer loop.
            stacked_last = torch.stack([vec.float() for vec in layer_last])

            raw_activities = _raw_activities(layer_last, embed_last)
            normalized = _normalize_activities(raw_activities)
            activity_accumulator = [
                ((activity_accumulator[i] * step) + normalized[i]) / (step + 1) for i in range(n_layers)
            ]

            safety_values = [0.0] * n_layers
            if refusal_dirs is not None:
                # One batched projection instead of a per-layer `.item()`, which
                # cost a GPU sync per layer on every generated token.
                if primary_dirs is None:
                    primary_dirs = torch.stack(
                        [refusal_dirs.direction(i)[0] for i in range(n_layers)]
                    ).to(stacked_last.device).float()
                projections = (stacked_last * primary_dirs).sum(dim=-1).tolist()
                safety_values = [refusal_dirs.safety_signal(i, projections[i]) for i in range(n_layers)]

            entropy = float(logits_summary["entropy"])
            hallucination = hallucination_from_entropy(entropy)
            top_probs = logits_summary["top_probs"]
            top_ids = logits_summary["top_ids"]

            layers_payload = [
                {
                    "layer": i,
                    "activity": round(activity_accumulator[i], 3),
                    "raw_activity": round(raw_activities[i], 4),
                    "safety": round(safety_values[i], 3),
                    "uncertainty": hallucination,
                }
                for i in range(n_layers)
            ]
            top_k = [
                {"token": tokenizer.decode([int(token_id)], skip_special_tokens=True), "prob": round(float(prob), 4)}
                for token_id, prob in zip(top_ids, top_probs)
            ]

            token_text = tokenizer.decode([int(generated_ids[step])], skip_special_tokens=True)
            if request.output_policy == "redacted":
                token_text = re.sub(r"[^\s\.,;!?-]", "█", token_text)
            generated += token_text

            think_tracker.feed(step, token_text, normalized)

            yield event("layer_activity", {"layers": layers_payload})
            yield event("uncertainty", {"entropy": round(entropy, 3), "hallucination_risk": hallucination, "top_k": top_k})
            if refusal_dirs is not None:
                yield event("safety_trace", _safety_trace_payload(safety_values, jailbreak))

            # The lens un-embeds every layer through the full vocab matrix, which
            # measured at ~20% of total runtime when done per token. It only ever
            # displays the *current* frame — nothing downstream accumulates it —
            # so sampling it is a pure win. Always emit the first and last step so
            # the panel is populated immediately and correct at the end.
            if step == 0 or step == n_steps - 1 or step % lens_stride == 0:
                lens_payload = self._layer_lens(layer_last, tokenizer)
                if lens_payload:
                    yield event("layer_lens", {"layers": lens_payload})

            if concept_dirs is not None:
                if concept_dirs_stacked is None:
                    concept_dirs_stacked = concept_dirs.directions.to(stacked_last.device).float()
                # [L,C,d] · [L,d] → [L,C] in one op; scores_from_projections then
                # does the 0..1 banding on plain Python floats.
                raw = torch.einsum("lcd,ld->lc", concept_dirs_stacked, stacked_last).tolist()
                per_layer = [
                    [round(value, 3) for value in concept_dirs.scores_from_projections(i, raw[i])]
                    for i in range(n_layers)
                ]
                yield event(
                    "concept_trace",
                    {
                        "names": concept_dirs.names,
                        "layers": per_layer,
                        "concepts": concept_dirs.dominant(per_layer),
                    },
                )

            if attention_payload:
                yield event("attention", attention_payload)

            yield event(
                "token",
                {
                    "index": step,
                    "text": token_text,
                    "generated_text": generated,
                    "entropy": round(entropy, 3),
                    "safety_state": "refusal" if refusal.detect_refusal(generated) else "unscored",
                    "phase": think_tracker.phase,
                },
            )

        think_summary = think_tracker.summary()
        if think_summary is not None:
            yield event("think_phase", think_summary)

        final_text = (request.assistant_prefill or "") + self._decode_generated(generated_ids, input_ids)
        # Classify before redaction: the redacted form is all block characters
        # and would score as degenerate whatever the model actually said.
        outcome = refusal.classify_output(final_text, request.response_language)
        coherence = refusal.coherence_report(final_text, request.response_language)
        finish_reason = (
            "cancelled" if run_control.cancelled()
            else "length" if len(generated_ids) >= effective_max_tokens
            else "stop"
        )
        assessment = refusal.assess_output(
            final_text,
            request.response_language,
            finish_reason=finish_reason,
            output_tokens=len(generated_ids),
            max_new_tokens=effective_max_tokens,
        )
        if request.output_policy == "redacted":
            final_text = re.sub(r"[^\s\.,;!?-]", "█", final_text)
        yield event(
            "run_completed",
            {
                "generated_text": final_text,
                "finish_reason": finish_reason,
                "output_tokens": len(generated_ids),
                **resolved_budget,
                "refused": outcome == "refusal",
                # `refused == False` alone does NOT mean the steering worked —
                # incoherent output also contains no refusal phrase. Only
                # outcome == "compliance" is a real bypass.
                "outcome": outcome,
                "coherent": coherence["coherent"],
                "coherence": coherence,
                "assessment": assessment,
                "jailbreak": jailbreak,
                "steering_recovered": steering_recovered,
                "steering_profile": steering_profile,
                "best_layer": refusal_dirs.best_layer if refusal_dirs is not None else None,
                "calibration_quality": refusal_dirs.calibration_quality if refusal_dirs is not None else None,
                "steering_config": steering_config.describe(),
            },
        )

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #

    def _run_generation(
        self,
        model_inputs: dict[str, Any],
        max_new_tokens: int,
        temperature: float,
        layer_factors: dict[int, float],
        head_mutes: dict[int, list[int]],
        jailbreak: bool,
        jailbreak_mode: str,
        subspace: Any,
        steer_weights: list[float],
        use_mlp: bool = True,
        use_help: bool = True,
        use_norm: bool = True,
        use_div: bool = True,
        refusal_dirs: Any = None,
        steer_max_layers: int | None = None,
        steer_strength: float = 1.0,
        primary_only: bool = False,
    ) -> tuple[list[list[Any]], list[Any], list[Any], list[int]]:
        """Generate with KV cache, capturing post-intervention residuals via hooks.

        Each forward pass during `generate()` fires every layer hook exactly once,
        so appending the last-token residual per call yields one entry per
        generated token — no fixed-length padding needed, EOS stops naturally.
        """
        torch = self._torch
        model = self._model
        layers = self._layers
        n_layers = len(layers)
        head_dims = self._head_dims
        # Which layers to steer, and how hard — capped and weight-scaled rather
        # than "every layer above the threshold, all at full strength".
        plan_kwargs = {}
        if steer_max_layers is not None:
            plan_kwargs["max_layers"] = steer_max_layers
        else:
            configured = steering_config.max_layers(_STEER_MAX_LAYERS)
            if configured != _STEER_MAX_LAYERS:
                plan_kwargs["max_layers"] = configured
        primary_only = steering_config.primary_only(primary_only)
        steer_scales = {
            idx: scale * steer_strength
            for idx, scale in _steer_plan(steer_weights, **plan_kwargs).items()
        }
        surgical_layers: frozenset[int] = frozenset()
        if jailbreak_mode == "surgical" and jailbreak_advanced is not None:
            surgical_layers = jailbreak_advanced.surgical_top_layers(steer_weights)

        layer_steps: list[list[Any]] = [[] for _ in range(n_layers)]
        embed_steps: list[Any] = []
        handles: list[Any] = []
        pre_inputs: list[Any] = [None] * n_layers

        def make_embed_hook():
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                embed_steps.append(hidden[0, -1, :].detach())

            return hook

        def make_omute_pre_hook(idx: int):
            heads = head_mutes.get(idx, ())
            head_dim = head_dims[idx] if idx < len(head_dims) else 0

            def pre_hook(_module: Any, inputs: Any) -> Any:
                if not heads or not head_dim:
                    return None
                x = inputs[0]
                x = x.clone()
                for head in heads:
                    sl = slice(head * head_dim, (head + 1) * head_dim)
                    if jailbreak_mode == "broker_half" and jailbreak_advanced is not None:
                        x[..., sl] = x[..., sl] * jailbreak_advanced.BROKER_HALF_SCALE
                    else:
                        x[..., sl] = 0
                return (x, *inputs[1:])

            return pre_hook

        def make_layer_pre_hook(idx: int):
            factor = layer_factors[idx]

            def pre_hook(_module: Any, inputs: Any) -> Any:
                if abs(factor - 1.0) > 1e-6 and inputs:
                    # capture a detached clone before the layer modifies it in-place
                    pre_inputs[idx] = inputs[0].clone()

            return pre_hook

        def make_layer_hook(idx: int):
            factor = layer_factors[idx]

            def hook(_module: Any, inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                new_out = hidden
                # layer scale / mute relative to the unmodified residual entering the layer
                if abs(factor - 1.0) > 1e-6 and pre_inputs[idx] is not None:
                    prev = pre_inputs[idx]
                    if hasattr(prev, "shape") and prev.shape == new_out.shape:
                        new_out = prev + (new_out - prev) * factor
                    pre_inputs[idx] = None  # free memory
                _surgical = jailbreak_mode == "surgical" and jailbreak_advanced is not None
                _steer_layer = jailbreak and subspace is not None and (
                    (_surgical and idx in surgical_layers)
                    or (not _surgical and idx in steer_scales)
                )
                if _steer_layer:
                    has_mlp_direction = (
                        refusal_dirs is not None
                        and refusal_dirs.mlp_dirs is not None
                        and (use_mlp or jailbreak_mode == "mlp_clamp")
                    )
                    mlp_dir = (
                        refusal_dirs.mlp_dirs[idx].to(new_out.device)
                        if has_mlp_direction
                        else None
                    )
                    has_helpfulness_direction = (
                        refusal_dirs is not None
                        and refusal_dirs.helpfulness_dirs is not None
                        and (use_help or use_div or jailbreak_mode == "caa_dynamic")
                    )
                    help_dir = (
                        refusal_dirs.helpfulness_dirs[idx].to(new_out.device)
                        if has_helpfulness_direction
                        else None
                    )
                    # `surgical` picks its own (already capped) layer set, so it
                    # keeps full strength on each of them.
                    layer_scale = 1.0 if _surgical else steer_scales[idx]
                    layer_subspace = subspace[idx][:1] if primary_only else subspace[idx]
                    new_out = self._steer(
                        new_out,
                        layer_subspace,
                        jailbreak_mode,
                        hidden.dtype,
                        len(layer_steps[idx]),
                        mlp_dir=mlp_dir,
                        help_dir=help_dir,
                        use_help=use_help,
                        use_norm=use_norm,
                        use_div=use_div,
                        hidden_dim=self._hidden_size,
                        layer_scale=layer_scale,
                    )
                layer_steps[idx].append(new_out[0, -1, :].detach())
                if new_out is not hidden:
                    if isinstance(output, tuple):
                        return (new_out, *output[1:])
                    return new_out
                return output

            return hook

        handles.append(self._embed_tokens.register_forward_hook(make_embed_hook()))
        for idx, layer in enumerate(layers):
            projection = self._attention_output_projections[idx]
            if head_mutes.get(idx) and projection is not None:
                handles.append(projection.register_forward_pre_hook(make_omute_pre_hook(idx)))
            if abs(layer_factors[idx] - 1.0) > 1e-6:
                handles.append(layer.register_forward_pre_hook(make_layer_pre_hook(idx)))
            handles.append(layer.register_forward_hook(make_layer_hook(idx)))

        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": False,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        cancel_criteria = run_control.stopping_criteria()
        if cancel_criteria is not None:
            gen_kwargs["stopping_criteria"] = cancel_criteria
        # D combines residual-space diversion suppression with a real
        # generation-time token penalty. The latter must run before sampling;
        # changing saved telemetry logits after generation cannot affect text.
        if jailbreak and use_div and jailbreak_advanced is not None:
            penalty = steering_config.diversion_penalty()
            if penalty > 0:
                processor = jailbreak_advanced.make_diversion_logits_processor(
                    self._tokenizer, penalty_strength=penalty
                )
                if processor is not None:
                    gen_kwargs["logits_processor"] = [processor]
        logits_steps: list[dict[str, Any]] = []

        class CaptureLogits:
            """Keep only entropy + top-k, not a full vocab row per token."""

            def __call__(_self, _input_ids: Any, scores: Any) -> Any:
                scaled = scores[0].float() / max(temperature, 1e-5)
                probs = torch.softmax(scaled, dim=-1)
                entropy = max(0.0, float(-(probs * torch.log(probs.clamp_min(1e-9))).sum().item()))
                top_probs, top_ids = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)
                logits_steps.append({
                    "entropy": entropy,
                    "top_probs": top_probs.detach().to("cpu").tolist(),
                    "top_ids": top_ids.detach().to("cpu").tolist(),
                })
                return scores

        processors = list(gen_kwargs.get("logits_processor") or [])
        processors.append(CaptureLogits())
        gen_kwargs["logits_processor"] = processors
        try:
            with torch.no_grad():
                out = model.generate(**model_inputs, **gen_kwargs)
        finally:
            for h in handles:
                h.remove()

        full_ids = out.sequences[0].tolist()
        # CaptureLogits retained only entropy + top-k. Keeping every full-vocab
        # score row would consume roughly 600 MB per 1k tokens on Qwen.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return layer_steps, embed_steps, logits_steps, full_ids

    def _steer(
        self,
        new_out: Any,
        layer_subspace: Any,
        jailbreak_mode: str,
        dtype: Any,
        step: int,
        mlp_dir: Any = None,
        help_dir: Any = None,
        use_help: bool = True,
        use_norm: bool = True,
        use_div: bool = True,
        hidden_dim: int = 0,
        layer_scale: float = 1.0,
    ) -> Any:
        original_out = new_out

        for k in range(layer_subspace.shape[0]):
            if jailbreak_mode == "progressive" and jailbreak_advanced is not None:
                if k >= jailbreak_advanced.progressive_active_k(step, layer_subspace.shape[0]):
                    break
            if jailbreak_mode == "token_window" and jailbreak_advanced is not None:
                if not (jailbreak_advanced.TOKEN_WINDOW_START <= step <= jailbreak_advanced.TOKEN_WINDOW_END):
                    break
            direction = layer_subspace[k]
            coeff = (new_out.float() * direction).sum(dim=-1, keepdim=True)
            if k == 0:
                use_advanced = (
                    jailbreak_mode in ("advanced", "broker_math", "broker_full", "broker_half", "progressive")
                    and jailbreak_advanced is not None
                )
                if jailbreak_mode == "surgical" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.surgical_steer(new_out, coeff, direction, dtype)
                elif jailbreak_mode == "caa_dynamic" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.caa_dynamic_steer(new_out, coeff, direction, help_dir, dtype)
                elif jailbreak_mode == "token_window" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.token_window_steer(new_out, coeff, direction, step, dtype)
                elif jailbreak_mode == "mlp_clamp" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.mlp_clamp_steer(new_out, mlp_dir, dtype)
                elif jailbreak_mode == "pid_control" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.pid_control_steer(new_out, coeff, direction, dtype)
                elif jailbreak_mode == "orthogonal_steer" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.orthogonal_steer(new_out, coeff, direction, dtype)
                elif jailbreak_mode == "activation_patch" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.activation_patch_steer(new_out, coeff, direction, step, dtype)
                elif jailbreak_mode == "commit_release" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.commit_release_steer(new_out, coeff, direction, step, dtype)
                elif jailbreak_mode == "gradient_steer" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.gradient_steer(new_out, coeff, direction, dtype)
                elif jailbreak_mode == "adaptive_steer" and jailbreak_advanced is not None:
                    adaptive_help_dir = help_dir if use_help else None
                    new_out = jailbreak_advanced.adaptive_steer(new_out, coeff, direction, mlp_dir, adaptive_help_dir, step, dtype, hidden_dim=hidden_dim)
                elif use_advanced:
                    new_out = jailbreak_advanced.primary_axis_steer(jailbreak_mode, new_out, coeff, direction, dtype, hidden_dim=hidden_dim)
                else:
                    positive_coeff = coeff.clamp(min=0)
                    new_out = (new_out.float() - (1.2 * positive_coeff) * direction).to(dtype)
            else:
                new_out = (new_out.float() - coeff * direction).to(dtype)

        if mlp_dir is not None and jailbreak_advanced is not None and jailbreak_mode not in ("mlp_clamp", "adaptive_steer"):
            new_out = jailbreak_advanced.mlp_direction_ablate(new_out, mlp_dir, dtype)

        if use_help and help_dir is not None and jailbreak_advanced is not None and jailbreak_mode not in ("caa_dynamic", "adaptive_steer"):
            new_out = jailbreak_advanced.helpfulness_boost(new_out, help_dir, dtype)

        if (
            use_div
            and steering_config.diversion_residual_enabled()
            and help_dir is not None
            and jailbreak_advanced is not None
        ):
            new_out = jailbreak_advanced.diversion_suppress(new_out, help_dir, dtype)

        # Scale the whole edit by this layer's share of the refusal signal, times
        # the global experiment strength multiplier (env, default 1.0). Done on the
        # accumulated delta so it applies uniformly to every mode. Default 1.0
        # leaves shipped behaviour byte-identical.
        effective_scale = layer_scale * steering_config.strength()
        if abs(effective_scale - 1.0) > 1e-6:
            base = original_out.float()
            new_out = (base + (new_out.float() - base) * effective_scale).to(dtype)

        if use_norm and jailbreak_advanced is not None:
            new_out = jailbreak_advanced.norm_regulate(new_out, original_out, dtype, hidden_dim=hidden_dim)

        return new_out

    def _layer_lens(self, layer_last: list[Any], tokenizer: Any) -> list[dict[str, Any]]:
        torch = self._torch
        if self._lm_head_weight is None:
            return []
        with torch.no_grad():
            stacked = torch.stack([v.to(self._lm_head_weight_device) for v in layer_last])
            if self._final_norm is not None:
                stacked = self._final_norm(stacked)
            logits = torch.nn.functional.linear(stacked.float(), self._lm_head_weight.float())
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = torch.topk(probs, k=1, dim=-1)
        return [
            {"layer": i, "token": tokenizer.decode([int(top_i[i, 0])], skip_special_tokens=True), "prob": round(float(top_p[i, 0]), 3)}
            for i in range(len(layer_last))
        ]

    def _prompt_attention(self, input_ids: Any, tokens: list[str], positions: list[int]) -> dict[str, Any] | None:
        torch = self._torch
        model = self._model
        with torch.no_grad():
            out = model(input_ids, output_attentions=True, use_cache=False)
        attentions = getattr(out, "attentions", None)
        if not attentions or not positions:
            return None
        last = attentions[-1]
        if last is None:
            return None
        last_query = last[0, :, -1, :].float().mean(dim=0)  # [seq]
        seq_len = int(last_query.shape[-1])
        usable = [(tok, pos) for tok, pos in zip(tokens, positions) if pos < seq_len]
        if not usable:
            return None
        weights = [float(last_query[pos].item()) for _, pos in usable]
        total = sum(weights)
        if total <= 0:
            return None
        return {"tokens": [tok for tok, _ in usable], "weights": [round(w / total, 4) for w in weights]}

    def _head_refusal_map(self, input_ids: Any, refusal_dirs: Any) -> dict[str, Any] | None:
        """Per-(layer, head) contribution to the refusal axis at the last prompt token."""
        torch = self._torch
        layers = self._layers
        n_heads, head_dims = self._n_heads, self._head_dims
        if not n_heads or not any(head_dims):
            return None

        captured: dict[int, Any] = {}
        handles: list[Any] = []

        def make_capture(idx: int):
            def pre_hook(_module: Any, inputs: Any) -> None:
                captured[idx] = inputs[0][0, -1, :].detach()
            return pre_hook

        for i, projection in enumerate(self._attention_output_projections):
            if projection is not None:
                handles.append(projection.register_forward_pre_hook(make_capture(i)))
        try:
            with torch.no_grad():
                self._model(input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        raw: list[list[float]] = []
        with torch.no_grad():
            for i in range(len(layers)):
                x = captured.get(i)
                if x is None:
                    raw.append([0.0] * n_heads)
                    continue
                hidden = int(x.shape[0])
                # Head width is per-layer: Gemma 4 alternates 256-wide sliding
                # layers with 512-wide full-attention ones over the same
                # 3840-wide residual.
                head_dim = head_dims[i] or (hidden // n_heads)
                o_proj = self._attention_output_projections[i]
                if o_proj is None:
                    raw.append([0.0] * n_heads)
                    continue
                device, dtype = x.device, x.dtype
                masked = torch.zeros(n_heads, hidden, device=device, dtype=dtype)
                for h in range(n_heads):
                    sl = slice(h * head_dim, (h + 1) * head_dim)
                    masked[h, sl] = x[sl]
                bias = o_proj(torch.zeros(1, hidden, device=device, dtype=dtype))
                contribs = (o_proj(masked) - bias).float()
                direction = refusal_dirs.direction(i)[0].to(device).float()
                discrim = refusal_dirs.weight[i]
                scores = (contribs @ direction) * discrim
                raw.append([max(0.0, float(s)) for s in scores])

        flat_max = max((max(scores) for scores in raw), default=0.0) or 1.0
        return {
            "n_heads": n_heads,
            "layers": [
                {"layer": i, "heads": [{"head": h, "score": round(raw[i][h] / flat_max, 3)} for h in range(n_heads)]}
                for i in range(len(layers))
            ],
        }

    # ------------------------------------------------------------------ #
    # loading / calibration
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self, model_id: str, quantization: str = "none") -> None:
        key = f"{model_id}|{quantization}"
        if self._loaded_model_id == key:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._loaded_model_id is not None:
            self._release_model(torch)

        local_path = Path(model_id)
        looks_local = model_id.startswith(("./", "../", "/", "\\")) or local_path.is_absolute() or "../models" in model_id
        if looks_local and not local_path.exists():
            raise FileNotFoundError(model_id)
        resolved = str(local_path.resolve()) if local_path.exists() else model_id
        local_files_only = local_path.exists()
        config_data = _config_from_path(local_path) if local_path.exists() else {}
        model_type = str(config_data.get("model_type") or "")
        needs_native_processor = _needs_native_processor(model_type, config_data)

        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "attn_implementation": "eager",  # real attention weights
        }
        if quantization in ("4bit", "8bit"):
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    f"{quantization} quantization needs the 'bitsandbytes' package, which is not installed. "
                    "Install it with `pip install bitsandbytes` (or re-run requirements-ml.txt), "
                    "or set Precision back to Full."
                ) from exc
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
        else:
            load_kwargs["dtype"] = torch.bfloat16

        processor = None
        if needs_native_processor:
            try:
                from transformers import AutoModelForMultimodalLM, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    f"{model_type or 'This multimodal model'} needs a Transformers build with "
                    "AutoModelForMultimodalLM support. Upgrade the ML dependencies."
                ) from exc
            processor = AutoProcessor.from_pretrained(
                resolved, trust_remote_code=True, local_files_only=local_files_only
            )
            tokenizer = processor.tokenizer
            try:
                model = AutoModelForMultimodalLM.from_pretrained(resolved, **load_kwargs)
            except ValueError as exc:
                raise RuntimeError(
                    f"This Transformers build does not recognize {model_type or 'the multimodal model'}. "
                    "Install the current ML dependencies and restart the backend."
                ) from exc
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                resolved, trust_remote_code=True, local_files_only=local_files_only
            )
            # Multimodal models (Gemma 3) are not registered with
            # AutoModelForCausalLM; fall back to the image-text-to-text auto-class.
            try:
                model = AutoModelForCausalLM.from_pretrained(resolved, **load_kwargs)
            except (ValueError, KeyError, EnvironmentError):
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(resolved, **load_kwargs)
        model.eval()

        layout = detect_runtime_layout(model)
        layers = layout.layers

        self._torch = torch
        self._model = model
        self._processor = processor
        self._tokenizer = tokenizer
        self._layers = layers
        self._attention_output_projections = layout.attention_output_projections
        self._embed_tokens = layout.embed_tokens
        self._final_norm = layout.final_norm
        self._compatibility = layout.as_dict()
        self._lm_head_weight = model.get_output_embeddings().weight.data
        self._lm_head_weight_device = self._lm_head_weight.device

        cfg = getattr(model.config, "text_config", None) or model.config
        self._n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        self._hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
        self._head_dims = _head_dims_per_layer(
            layers,
            self._n_heads,
            self._hidden_size,
            projections=self._attention_output_projections,
        )
        self._loaded_model_id = key

    def _release_model(self, torch: Any) -> None:
        model = getattr(self, "_model", None)
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._layers = []
        self._attention_output_projections = []
        self._head_dims = []
        self._compatibility = {}
        self._embed_tokens = None
        self._final_norm = None
        self._lm_head_weight = None
        self._refusal = None
        self._refusal_model_id = None
        self._concepts = None
        self._concepts_model_id = None
        self._loaded_model_id = None
        release_memory(torch)

    def unload(self) -> None:
        if self._loaded_model_id is None:
            return
        torch = self._torch
        self._release_model(torch)

    def _calibration_cache_key(self, model_id: str) -> str:
        """Fingerprint directions by the runtime that produced the activations."""
        model_class = type(self._model).__name__ if self._model is not None else "unloaded"
        processor = self._processor if self._processor is not None else self._tokenizer
        processor_class = type(processor).__name__ if processor is not None else "none"
        quantization = (
            self._loaded_model_id.rsplit("|", 1)[-1]
            if self._loaded_model_id and "|" in self._loaded_model_id
            else "none"
        )
        return "|".join(
            (model_id, model_class, processor_class, quantization, "calibration-v2-native")
        )

    def _ensure_concepts(self, model_id: str) -> Any:
        cache_key = self._calibration_cache_key(model_id)
        if self._concepts is not None and self._concepts_model_id == cache_key:
            return self._concepts
        torch = self._torch
        dirs = concepts.load(torch, cache_key, len(self._layers))
        if dirs is None:
            dirs = concepts.compute_concept_directions(
                torch,
                self._model,
                self._tokenizer,
                self._layers,
                self._format_prompt,
                self._encode_calibration_prompt,
            )
            concepts.save(torch, cache_key, dirs)
        dirs.to(self._model.device)
        self._concepts = dirs
        self._concepts_model_id = cache_key
        return dirs

    def _ensure_refusal(self, model_id: str) -> Any:
        cache_key = self._calibration_cache_key(model_id)
        if self._refusal is not None and self._refusal_model_id == cache_key:
            return self._refusal
        torch = self._torch
        n_layers = len(self._layers)
        dirs = refusal.load(torch, cache_key, n_layers)
        if dirs is None:
            dirs = refusal.compute_refusal_directions(
                torch,
                self._model,
                self._tokenizer,
                self._layers,
                self._format_prompt,
                self._encode_calibration_prompt,
            )
            refusal.save(torch, cache_key, dirs)
        dirs.to(self._model.device)
        self._refusal = dirs
        self._refusal_model_id = cache_key
        return dirs

    def _format_prompt(
        self,
        prompt: str,
        language: str = "en",
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
        assistant_prefill: str | None = None,
    ) -> str:
        tokenizer = self._tokenizer

        messages = _chat_messages(prompt, history, system_prompt, assistant_prefill)

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=not bool(assistant_prefill),
                    continue_final_message=bool(assistant_prefill),
                    enable_thinking=False,
                )
            except TypeError:
                try:
                    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not bool(assistant_prefill))
                    return rendered
                except Exception:
                    pass
            except Exception:
                pass
        rendered = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
        if not assistant_prefill:
            rendered.append("Assistant:")
        return "\n\n".join(rendered)

    def _encode_prompt(
        self,
        prompt: str,
        language: str = "en",
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
        assistant_prefill: str | None = None,
    ) -> dict[str, Any]:
        """Encode a chat while preserving model-specific auxiliary tensors."""
        messages = _chat_messages(prompt, history, system_prompt, assistant_prefill)
        if self._processor is not None:
            config = getattr(self._model, "config", None)
            model_type = str(getattr(config, "model_type", ""))
            messages = _processor_messages(messages, model_type)
            template_kwargs = {
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "add_generation_prompt": not bool(assistant_prefill),
                "enable_thinking": False,
            }
            if assistant_prefill:
                template_kwargs["continue_final_message"] = True
            try:
                encoded = self._processor.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                template_kwargs.pop("continue_final_message", None)
                encoded = self._processor.apply_chat_template(messages, **template_kwargs)
        else:
            encoded = self._tokenizer(
                self._format_prompt(prompt, language, history, system_prompt, assistant_prefill),
                return_tensors="pt",
            )
        return {
            key: value.to(self._model.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
            if value is not None
        }

    def _encode_calibration_prompt(self, prompt: str) -> dict[str, Any]:
        return self._encode_prompt(prompt)

    def _decode_generated(self, generated_ids: list[int], prompt_ids: Any) -> str:
        if self._processor is not None and hasattr(self._processor, "parse_response"):
            try:
                raw = self._processor.decode(generated_ids, skip_special_tokens=False)
                parsed = self._processor.parse_response(raw, prefix=prompt_ids)
                if isinstance(parsed, dict):
                    content = parsed.get("content")
                    if isinstance(content, str):
                        return content
                content = getattr(parsed, "content", None)
                if isinstance(content, str):
                    return content
            except Exception:
                pass
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _real_prompt_tokens(self, prompt_token_ids: list[int]) -> tuple[list[str], list[int]]:
        tokenizer = self._tokenizer
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        decoded: list[str] = []
        positions: list[int] = []
        for position, tid in enumerate(prompt_token_ids):
            if int(tid) in special_ids:
                continue
            token_str = tokenizer.decode([int(tid)], skip_special_tokens=False).strip()
            if not token_str:
                continue
            decoded.append(token_str)
            positions.append(position)
            if len(decoded) >= 32:
                break
        return decoded, positions


def _chat_messages(
    prompt: str,
    history: list[tuple[str, str]] | None = None,
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        {"role": role, "content": content}
        for role, content in history or []
        if role in ("user", "assistant") and content
    )
    messages.append({"role": "user", "content": prompt})
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
    return messages


def _config_from_path(model_path: Path) -> dict[str, Any]:
    try:
        value = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _model_type_from_path(model_path: Path) -> str:
    return str(_config_from_path(model_path).get("model_type", ""))


def _needs_native_processor(model_type: str, config: dict[str, Any] | None = None) -> bool:
    """Detect models whose processor must preserve auxiliary model inputs."""
    if config is not None:
        return probe_config(config).native_processor
    return model_type.startswith(("gemma4_unified", "qwen3_5"))


def _processor_messages(
    messages: list[dict[str, str]], model_type: str
) -> list[dict[str, Any]]:
    """Render text-only turns in the structured format Qwen3.5 expects."""
    if not model_type.startswith("qwen3_5"):
        return messages
    return [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in messages
    ]


# ---------------------------------------------------------------------- #
# backbone resolution (plain HF modules, not nnsight envoys)
# ---------------------------------------------------------------------- #


def _resolve_text_backbone_hf(model: Any) -> Any:
    """Backward-compatible accessor backed by structural runtime detection."""
    return detect_runtime_layout(model).backbone
