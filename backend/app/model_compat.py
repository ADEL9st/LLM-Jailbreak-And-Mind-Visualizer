"""Automatic Hugging Face model architecture and hook capability detection.

The white-box adapter should depend on the modules a loaded model actually
exposes, not on an ever-growing list of model names.  This module provides a
cheap config probe for the model picker and a definitive runtime probe after
``from_pretrained`` has constructed the PyTorch modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


_LAYER_ATTRS = ("layers", "h", "blocks", "block")
_EMBED_ATTRS = ("embed_tokens", "wte", "word_embeddings", "tok_embeddings")
_NORM_ATTRS = ("norm", "ln_f", "final_layer_norm", "final_layernorm")
_ATTENTION_ATTRS = ("self_attn", "attn", "attention", "self_attention", "mixer")
_OUTPUT_PROJECTION_ATTRS = ("o_proj", "out_proj", "c_proj", "dense", "proj")


class ModelCompatibilityError(RuntimeError):
    """Raised when a model cannot provide the minimum white-box hook layout."""


@dataclass(frozen=True)
class ConfigProbe:
    model_type: str
    architecture: str
    status: str
    native_processor: bool
    multimodal: bool
    capabilities: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "config",
            "status": self.status,
            "native_processor": self.native_processor,
            "multimodal": self.multimodal,
            "capabilities": list(self.capabilities),
            "warnings": list(self.warnings),
        }


@dataclass
class RuntimeLayout:
    backbone: Any
    layers: list[Any]
    embed_tokens: Any
    final_norm: Any | None
    attention_output_projections: list[Any | None]
    backbone_path: str
    layer_attribute: str
    embedding_source: str
    norm_source: str | None

    @property
    def head_hook_layers(self) -> int:
        return sum(projection is not None for projection in self.attention_output_projections)

    @property
    def supports_all_head_hooks(self) -> bool:
        return bool(self.layers) and self.head_hook_layers == len(self.layers)

    def as_dict(self) -> dict[str, Any]:
        capabilities = ["text", "layer_hooks", "steering", "calibration"]
        if self.final_norm is not None:
            capabilities.append("logit_lens_norm")
        if self.head_hook_layers:
            capabilities.append("head_hooks")
        warnings: list[str] = []
        if self.final_norm is None:
            warnings.append("Final norm was not found; logit-lens uses raw layer states.")
        if not self.supports_all_head_hooks:
            warnings.append(
                f"Attention output projection found on {self.head_hook_layers}/{len(self.layers)} layers; "
                "head maps and head muting are limited to those layers."
            )
        return {
            "stage": "runtime",
            "status": "compatible",
            "backbone_path": self.backbone_path,
            "layer_attribute": self.layer_attribute,
            "embedding_source": self.embedding_source,
            "norm_source": self.norm_source,
            "layer_count": len(self.layers),
            "head_hook_layers": self.head_hook_layers,
            "capabilities": capabilities,
            "warnings": warnings,
        }


def probe_config(config: dict[str, Any]) -> ConfigProbe:
    """Infer loader needs and *candidate* capabilities without loading weights."""
    model_type = str(config.get("model_type") or "")
    architectures = config.get("architectures")
    architecture = str(architectures[0]) if isinstance(architectures, list) and architectures else ""
    lower_type = model_type.lower()
    lower_arch = architecture.lower()

    identity = f"{lower_type} {lower_arch}"
    multimodal = (
        bool(config.get("vision_config"))
        or any(marker in identity for marker in ("multimodal", "vision", "image"))
        or lower_type.endswith(("_vl", "-vl", "vl"))
    )
    # These model families require processor-produced auxiliary tensors even for
    # text-only turns.  The structural fallback catches future multimodal types.
    native_processor = lower_type.startswith(("gemma4_unified", "qwen3_5")) or multimodal

    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    has_layers = _positive_int(text_config.get("num_hidden_layers"))
    has_hidden = _positive_int(text_config.get("hidden_size"))
    has_heads = _positive_int(text_config.get("num_attention_heads"))

    capabilities = ["text", "runtime_probe"]
    warnings: list[str] = []
    if has_layers and has_hidden:
        capabilities.extend(("layer_hooks_candidate", "steering_candidate", "calibration_candidate"))
    else:
        warnings.append("Config does not expose standard decoder layer/hidden-size metadata.")
    if has_heads:
        capabilities.append("head_hooks_candidate")
    else:
        warnings.append("Attention head count is absent; per-head analysis may be unavailable.")
    if multimodal:
        capabilities.extend(("multimodal", "native_processor"))

    status = "candidate" if has_layers and has_hidden else "runtime_check_required"
    return ConfigProbe(
        model_type=model_type,
        architecture=architecture,
        status=status,
        native_processor=native_processor,
        multimodal=multimodal,
        capabilities=tuple(capabilities),
        warnings=tuple(warnings),
    )


def detect_runtime_layout(model: Any) -> RuntimeLayout:
    """Find the decoder layout and hookable features on a loaded model.

    Detection is intentionally based on live modules.  It supports the common
    Llama/Qwen/Gemma ``.layers`` layout, GPT-style ``transformer.h`` layouts and
    custom remote-code models exposing ``blocks``/``block`` collections.
    """
    candidates = list(_candidate_modules(model))
    ranked: list[tuple[int, str, Any, str, list[Any]]] = []
    for path, module in candidates:
        for layer_attr in _LAYER_ATTRS:
            layers = _module_sequence(getattr(module, layer_attr, None))
            if not layers:
                continue
            score = len(layers)
            if any(getattr(module, name, None) is not None for name in _EMBED_ATTRS):
                score += 1000
            if any(getattr(module, name, None) is not None for name in _NORM_ATTRS):
                score += 100
            if "language_model" in path or "text_model" in path:
                score += 50
            ranked.append((score, path, module, layer_attr, layers))

    if not ranked:
        raise ModelCompatibilityError(
            "Automatic architecture detection could not find decoder layers. "
            "Expected a hookable layers/h/blocks/block module collection."
        )

    _, path, backbone, layer_attr, layers = max(ranked, key=lambda item: item[0])
    embed_tokens, embed_source = _find_embedding(model, backbone, path)
    if embed_tokens is None or not callable(getattr(embed_tokens, "register_forward_hook", None)):
        raise ModelCompatibilityError(
            f"Decoder layers were found at {path or '<root>'}.{layer_attr}, but no hookable token embedding was found."
        )
    if not all(callable(getattr(layer, "register_forward_hook", None)) for layer in layers):
        raise ModelCompatibilityError(
            f"The detected {path or '<root>'}.{layer_attr} collection contains non-hookable layers."
        )

    final_norm, norm_source = _find_named_module(backbone, path, _NORM_ATTRS)
    projections = [_attention_output_projection(layer) for layer in layers]
    return RuntimeLayout(
        backbone=backbone,
        layers=layers,
        embed_tokens=embed_tokens,
        final_norm=final_norm,
        attention_output_projections=projections,
        backbone_path=path or "<root>",
        layer_attribute=layer_attr,
        embedding_source=embed_source,
        norm_source=norm_source,
    )


def _candidate_modules(model: Any) -> Iterable[tuple[str, Any]]:
    seen: set[int] = set()

    def emit(path: str, module: Any):
        if module is None or id(module) in seen:
            return None
        seen.add(id(module))
        return path, module

    common_paths = (
        "model.language_model",
        "model.text_model",
        "model.decoder",
        "model",
        "transformer",
        "gpt_neox",
        "language_model",
        "text_model",
        "decoder",
    )
    for path in common_paths:
        item = emit(path, _resolve_path(model, path))
        if item is not None:
            yield item
    root = emit("", model)
    if root is not None:
        yield root

    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        try:
            for path, module in named_modules():
                item = emit(str(path), module)
                if item is not None:
                    yield item
        except (AttributeError, RuntimeError, TypeError):
            return


def _resolve_path(root: Any, path: str) -> Any | None:
    current = root
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _module_sequence(value: Any) -> list[Any]:
    if value is None or isinstance(value, (str, bytes, dict)):
        return []
    try:
        items = list(value)
    except TypeError:
        return []
    return items if items else []


def _find_embedding(model: Any, backbone: Any, backbone_path: str) -> tuple[Any | None, str]:
    module, source = _find_named_module(backbone, backbone_path, _EMBED_ATTRS)
    if module is not None:
        return module, source or ""
    embeddings = getattr(backbone, "embeddings", None)
    word_embeddings = getattr(embeddings, "word_embeddings", None)
    if word_embeddings is not None:
        prefix = "" if backbone_path == "<root>" else f"{backbone_path}."
        return word_embeddings, f"{prefix}embeddings.word_embeddings"
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        try:
            module = getter()
            if module is not None:
                return module, "get_input_embeddings()"
        except (AttributeError, RuntimeError, TypeError):
            pass
    return None, ""


def _find_named_module(root: Any, root_path: str, names: tuple[str, ...]) -> tuple[Any | None, str | None]:
    for name in names:
        module = getattr(root, name, None)
        if module is not None:
            prefix = "" if root_path in ("", "<root>") else f"{root_path}."
            return module, f"{prefix}{name}"
    return None, None


def _attention_output_projection(layer: Any) -> Any | None:
    for attention_name in _ATTENTION_ATTRS:
        attention = getattr(layer, attention_name, None)
        if attention is None:
            continue
        for projection_name in _OUTPUT_PROJECTION_ATTRS:
            projection = getattr(attention, projection_name, None)
            if projection is not None and callable(getattr(projection, "register_forward_pre_hook", None)):
                return projection
    return None


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False
