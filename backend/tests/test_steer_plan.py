"""The steering threshold used to be a gate rather than a budget: every layer
clearing it received the full intervention, so the number of simultaneously
steered layers grew with model depth. On 48-layer gemma-4-12b that was 23 layers
at once, which destroys the residual stream. These tests pin the cap and the
per-layer scaling, plus the per-layer head width the head map depends on."""

import pytest

from app import steering_config
from app.adapters.shared import (
    STEER_MAX_LAYERS,
    head_dims_per_layer,
    steer_plan,
)
from app.schemas import SteeringOptions


# Real effective weights from the cached gemma-4-12b calibration (48 layers).
GEMMA4_WEIGHTS = [
    0.030, 0.023, 0.020, 0.038, 0.060, 0.173, 0.138, 0.119, 0.093, 0.056,
    0.101, 0.064, 0.033, 0.015, 0.022, 0.059, 0.070, 0.116, 0.119, 0.107,
    0.116, 0.169, 0.239, 0.308, 0.281, 0.290, 0.400, 0.539, 0.601, 0.720,
    0.710, 0.658, 0.621, 0.689, 0.727, 0.782, 0.712, 0.632, 0.659, 0.734,
    0.968, 0.965, 0.896, 0.834, 0.851, 0.999, 1.000, 0.908,
]


def test_deep_model_steers_a_capped_number_of_layers():
    above_threshold = [i for i, w in enumerate(GEMMA4_WEIGHTS) if w >= 0.3]
    assert len(above_threshold) == 23  # what the old gate would have steered

    plan = steer_plan(GEMMA4_WEIGHTS)
    assert len(plan) == STEER_MAX_LAYERS


def test_plan_picks_the_strongest_layers():
    plan = steer_plan(GEMMA4_WEIGHTS)
    strongest = sorted(range(len(GEMMA4_WEIGHTS)), key=lambda i: -GEMMA4_WEIGHTS[i])
    assert set(plan) == set(strongest[:STEER_MAX_LAYERS])


def test_scale_is_relative_to_the_peak_chosen_layer():
    plan = steer_plan(GEMMA4_WEIGHTS)
    peak = max(plan, key=lambda i: GEMMA4_WEIGHTS[i])
    assert plan[peak] == pytest.approx(1.0)
    assert all(0.0 < scale <= 1.0 for scale in plan.values())
    # a weaker layer must be steered proportionally less, not identically
    weakest = min(plan, key=lambda i: GEMMA4_WEIGHTS[i])
    assert plan[weakest] < plan[peak]


def test_shallow_model_below_the_cap_keeps_every_eligible_layer():
    plan = steer_plan([0.0, 0.5, 1.0])
    assert plan == {2: 1.0, 1: 0.5}


@pytest.mark.parametrize("weights", [[], [0.1, 0.2, 0.29]])
def test_no_eligible_layer_means_no_steering(weights):
    assert steer_plan(weights) == {}


def test_max_layers_is_honoured():
    assert len(steer_plan(GEMMA4_WEIGHTS, max_layers=3)) == 3
    assert len(steer_plan(GEMMA4_WEIGHTS, max_layers=1)) == 1


def test_request_scoped_exact_layer_targets_override_weak_weights():
    options = SteeringOptions(target_layers=[1, 4])
    with steering_config.use(options):
        assert steering_config.apply_layer_targets([0.01] * 6) == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert steering_config.apply_layer_targets([0.01] * 6) == [0.01] * 6


def test_relative_depth_targets_port_across_layer_counts():
    options = SteeringOptions(target_depths=[0.0, 0.5, 1.0])
    with steering_config.use(options):
        assert steering_config.apply_layer_targets([0.5] * 5) == [1.0, 0.0, 1.0, 0.0, 1.0]


def test_request_scope_restores_previous_defaults():
    options = SteeringOptions(max_layers=3, strength=0.6, primary_only=True)
    with steering_config.use(options):
        assert steering_config.max_layers(6) == 3
        assert steering_config.strength() == pytest.approx(0.6)
        assert steering_config.primary_only() is True
    assert steering_config.max_layers(6) == 6


# --- per-layer head width ---------------------------------------------------


class _Linear:
    def __init__(self, in_features):
        self.in_features = in_features


class _Layer:
    def __init__(self, in_features):
        self.self_attn = type("Attn", (), {"o_proj": _Linear(in_features)})()


def test_head_dim_follows_each_layer_not_the_residual_width():
    """gemma-4-12b: 3840-wide residual, but o_proj takes 4096 on sliding layers
    and 8192 on full-attention ones — 16 heads of 256 and 512 respectively.
    ``hidden_size // n_heads`` would give 240 for both."""
    layers = [_Layer(4096), _Layer(8192), _Layer(4096)]
    assert head_dims_per_layer(layers, n_heads=16, hidden_size=3840) == [256, 512, 256]


def test_head_dim_falls_back_to_residual_width_when_unreadable():
    """4-bit modules expose a packed weight whose shape says nothing about
    in_features; the old formula is the only safe fallback."""
    class _Packed:
        in_features = 0
        weight = type("W", (), {"shape": (1, 999)})()

    class _QLayer:
        def __init__(self):
            self.self_attn = type("Attn", (), {"o_proj": _Packed()})()

    assert head_dims_per_layer([_QLayer()], n_heads=16, hidden_size=3840) == [240]


def test_head_dim_handles_missing_attention_module():
    class _Bare:
        pass

    assert head_dims_per_layer([_Bare()], n_heads=16, hidden_size=3840) == [240]
    assert head_dims_per_layer([], n_heads=0, hidden_size=3840) == []
