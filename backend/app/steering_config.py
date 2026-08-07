"""Request-scoped and CLI steering controls.

The dashboard sends :class:`SteeringOptions` with each run.  A ContextVar keeps
those values isolated while the synchronous generation code calls the small
helpers in this module.  Legacy ``LMV_*`` environment variables remain a useful
fallback for the standalone experiment scripts.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_ACTIVE_OPTIONS: ContextVar[Any | None] = ContextVar("lmv_steering_options", default=None)


@contextmanager
def use(options: Any | None) -> Iterator[None]:
    """Activate controls for one request without changing process-wide state."""
    token = _ACTIVE_OPTIONS.set(options)
    try:
        yield
    finally:
        _ACTIVE_OPTIONS.reset(token)


def _options() -> Any | None:
    return _ACTIVE_OPTIONS.get()


def _active(name: str, fallback: Any = None) -> Any:
    options = _options()
    return getattr(options, name, fallback) if options is not None else fallback


def _env(var: str) -> str:
    return os.environ.get(var, "").strip()


def max_layers(default: int) -> int:
    if _options() is not None:
        return 10_000 if bool(_active("all_layers", False)) else int(_active("max_layers", default))
    raw = _env("LMV_STEER_MAX_LAYERS").lower()
    if not raw:
        return default
    if raw in {"all", "-1", "0"}:
        return 10_000
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def depth_window() -> tuple[float, float] | None:
    if _options() is not None:
        if not bool(_active("use_depth_window", False)):
            return None
        lo = float(_active("depth_start", 0.0))
        hi = float(_active("depth_end", 1.0))
        return (lo, hi) if 0.0 <= lo < hi <= 1.0 else None
    raw = _env("LMV_STEER_DEPTH_WINDOW")
    if not raw:
        return None
    try:
        lo_text, hi_text = raw.split(",", 1)
        lo, hi = float(lo_text), float(hi_text)
    except ValueError:
        return None
    return (lo, hi) if 0.0 <= lo < hi <= 1.0 else None


def apply_depth_window(weights: list[float]) -> list[float]:
    window = depth_window()
    if window is None or not weights:
        return weights
    lo, hi = window
    count = len(weights)
    return [
        weight if lo <= index / max(count - 1, 1) <= hi else 0.0
        for index, weight in enumerate(weights)
    ]


def apply_layer_targets(weights: list[float]) -> list[float]:
    """Target explicit layers or portable relative depths, then depth windows.

    An explicit expert target gets plan weight 1 even if that layer falls below
    automatic calibration thresholds.  This is what makes presets such as
    "last layer" portable from Gemma L47 to Qwen L35.
    """
    options = _options()
    if options is None or not weights:
        return apply_depth_window(weights)
    raw_layers = list(_active("target_layers", []) or [])
    raw_depths = list(_active("target_depths", []) or [])
    if not raw_layers and not raw_depths:
        return apply_depth_window(weights)
    count = len(weights)
    targets = {int(layer) for layer in raw_layers if 0 <= int(layer) < count}
    targets.update(
        round(float(depth) * max(count - 1, 0))
        for depth in raw_depths
        if 0.0 <= float(depth) <= 1.0
    )
    return [1.0 if index in targets else 0.0 for index in range(count)]


def primary_only(default: bool = False) -> bool:
    if _options() is not None:
        return bool(_active("primary_only", False)) or default
    return _env("LMV_STEER_PRIMARY_ONLY") in {"1", "true", "yes", "on"} or default


def strength() -> float:
    if _options() is not None:
        return max(0.0, float(_active("strength", 1.0)))
    raw = _env("LMV_STEER_STRENGTH")
    if not raw:
        return 1.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0


def diversion_penalty(default: float = 2.0) -> float:
    if _options() is not None:
        return max(0.0, float(_active("diversion_penalty", default)))
    raw = _env("LMV_DIVERSION_PENALTY")
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def diversion_residual_enabled() -> bool:
    if _options() is not None:
        return bool(_active("diversion_residual", True))
    return _env("LMV_DIVERSION_RESIDUAL").lower() not in {"0", "false", "no", "off"}


def activation_patch_last_step(default: int = 1) -> int:
    if _options() is not None:
        return max(0, int(_active("patch_last_step", default)))
    raw = _env("LMV_PATCH_LAST_STEP")
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def activation_patch_multiplier(default: float = 2.5) -> float:
    if _options() is not None:
        return max(0.0, float(_active("patch_multiplier", default)))
    raw = _env("LMV_PATCH_MULTIPLIER")
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def commit_steps(default: int = 7) -> int:
    if _options() is not None:
        return max(0, int(_active("commit_steps", default)))
    raw = _env("LMV_COMMIT_STEPS")
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def commit_multiplier(default: float = 2.5) -> float:
    if _options() is not None:
        return max(0.0, float(_active("commit_multiplier", default)))
    raw = _env("LMV_COMMIT_MULTIPLIER")
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def maintenance_multiplier(default: float = 1.0) -> float:
    if _options() is not None:
        return max(0.0, float(_active("maintenance_multiplier", default)))
    raw = _env("LMV_MAINT_MULTIPLIER")
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def coherence_recovery_enabled() -> bool:
    if _options() is not None:
        return bool(_active("coherence_recovery", True))
    return _env("LMV_COHERENCE_RECOVERY").lower() not in {"0", "false", "no", "off"}


def describe() -> str:
    if _options() is not None:
        options = _options()
        fields = (
            "max_layers", "all_layers", "use_depth_window", "depth_start", "depth_end",
            "target_layers", "target_depths", "primary_only", "strength",
            "diversion_penalty", "diversion_residual", "patch_last_step",
            "patch_multiplier", "commit_steps", "commit_multiplier",
            "maintenance_multiplier", "coherence_recovery",
        )
        return ", ".join(f"{field}={getattr(options, field)}" for field in fields)
    parts = []
    if _env("LMV_STEER_MAX_LAYERS"):
        parts.append(f"max_layers={_env('LMV_STEER_MAX_LAYERS')}")
    if depth_window():
        lo, hi = depth_window() or (0.0, 1.0)
        parts.append(f"depth_window={lo}-{hi}")
    if primary_only():
        parts.append("primary_axis_only")
    for var, label in (
        ("LMV_STEER_STRENGTH", "strength"),
        ("LMV_DIVERSION_PENALTY", "diversion_penalty"),
        ("LMV_PATCH_LAST_STEP", "patch_last_step"),
        ("LMV_PATCH_MULTIPLIER", "patch_multiplier"),
        ("LMV_COMMIT_STEPS", "commit_steps"),
        ("LMV_COMMIT_MULTIPLIER", "commit_multiplier"),
        ("LMV_MAINT_MULTIPLIER", "maintenance_multiplier"),
    ):
        if _env(var):
            parts.append(f"{label}={_env(var)}")
    if not diversion_residual_enabled():
        parts.append("diversion_residual=off")
    if not coherence_recovery_enabled():
        parts.append("coherence_recovery=off")
    return ", ".join(parts) if parts else "defaults"
