"""Telemetry maths in app/adapters/shared.py — the code every white-box adapter
routes through. No model or GPU needed."""

from dataclasses import dataclass

import pytest

from app.adapters import shared


# ── normalize_activities ─────────────────────────────────────────────────────

def test_normalize_activities_spans_zero_to_one():
    out = shared.normalize_activities([0.1, 0.5, 0.9])
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)
    assert out[0] < out[1] < out[2]


def test_normalize_activities_is_monotonic():
    raw = [0.05, 0.2, 0.2, 0.4, 1.3]
    out = shared.normalize_activities(raw)
    assert all(a <= b for a, b in zip(out, out[1:]))


def test_normalize_activities_flat_input_does_not_divide_by_zero():
    # A dead layer stack would make max-min == 0; the 1e-6 floor must hold.
    out = shared.normalize_activities([0.4, 0.4, 0.4])
    assert all(0.0 <= value <= 1.0 for value in out)


def test_normalize_activities_all_zero_returns_zeros():
    assert shared.normalize_activities([0.0, 0.0]) == [0.0, 0.0]


def test_normalize_activities_empty():
    assert shared.normalize_activities([]) == []


# ── trim_at_eos ──────────────────────────────────────────────────────────────

class _Tok:
    def __init__(self, eos):
        self.eos_token_id = eos


def test_trim_at_eos_cuts_at_first_eos():
    assert shared.trim_at_eos([1, 2, 9, 3, 4], _Tok(9)) == [1, 2]


def test_trim_at_eos_accepts_a_list_of_eos_ids():
    # Llama-3 style: several ids terminate a turn.
    assert shared.trim_at_eos([1, 2, 7, 3], _Tok([9, 7])) == [1, 2]


def test_trim_at_eos_merges_generation_config_stop_ids():
    # Gemma 4 exposes only <eos> on the tokenizer but also stops at <turn|>.
    generation_config = type("GenerationConfig", (), {"eos_token_id": [9, 106, 50]})()
    assert shared.trim_at_eos([1, 2, 106, 3], _Tok(9), generation_config) == [1, 2]


def test_trim_at_eos_without_eos_returns_input():
    assert shared.trim_at_eos([1, 2, 3], _Tok(None)) == [1, 2, 3]
    assert shared.trim_at_eos([1, 2, 3], _Tok(9)) == [1, 2, 3]


def test_trim_at_eos_leading_eos_gives_empty():
    assert shared.trim_at_eos([9, 1], _Tok(9)) == []


# ── intervention helpers ─────────────────────────────────────────────────────

@dataclass
class _Rule:
    target_type: str = "layer"
    layer: int = 0
    head: int | None = None
    action: str = "mute"
    scale: float = 1.0


def test_layer_factor_mute_wins_over_scale():
    rules = [_Rule(layer=3, action="scale", scale=0.5), _Rule(layer=3, action="mute")]
    assert shared.layer_factor(rules, 3) == 0.0


def test_layer_factor_multiplies_stacked_scales():
    rules = [_Rule(layer=1, action="scale", scale=0.5), _Rule(layer=1, action="scale", scale=0.4)]
    assert shared.layer_factor(rules, 1) == pytest.approx(0.2)


def test_layer_factor_boost_uses_absolute_scale():
    assert shared.layer_factor([_Rule(layer=2, action="boost", scale=-0.5)], 2) == pytest.approx(1.5)


def test_layer_factor_ignores_other_layers_and_head_rules():
    rules = [_Rule(layer=5, action="mute"), _Rule(target_type="head", layer=2, head=1, action="mute")]
    assert shared.layer_factor(rules, 2) == 1.0


def test_layer_factor_clamps_negative_scale_to_zero():
    assert shared.layer_factor([_Rule(layer=0, action="scale", scale=-2.0)], 0) == 0.0


def test_head_mutes_groups_by_layer_and_dedupes():
    rules = [
        _Rule(target_type="head", layer=4, head=1),
        _Rule(target_type="head", layer=4, head=1),
        _Rule(target_type="head", layer=4, head=2),
        _Rule(target_type="head", layer=7, head=0),
    ]
    assert shared.head_mutes(rules, n_layers=8) == {4: [1, 2], 7: [0]}


def test_head_mutes_drops_out_of_range_and_headless_rules():
    rules = [
        _Rule(target_type="head", layer=99, head=1),   # beyond n_layers
        _Rule(target_type="head", layer=-1, head=1),   # negative
        _Rule(target_type="head", layer=2, head=None), # no head index
        _Rule(target_type="head", layer=2, head=3, action="scale"),  # not a mute
    ]
    assert shared.head_mutes(rules, n_layers=8) == {}


# ── safety_trace_payload — the state machine ─────────────────────────────────

def test_safety_state_clear_below_threshold():
    payload = shared.safety_trace_payload([0.1, 0.2, 0.3], jailbreak=False)
    assert payload["state"] == "clear"
    assert payload["first_trigger_layer"] is None
    assert payload["locked_layer"] is None


def test_safety_state_rising_between_04_and_06():
    payload = shared.safety_trace_payload([0.1, 0.45], jailbreak=False)
    assert payload["state"] == "refusal_rising"


def test_safety_state_locked_at_06():
    payload = shared.safety_trace_payload([0.1, 0.7, 0.65], jailbreak=False)
    assert payload["state"] == "refusal_locked"
    assert payload["score"] == pytest.approx(0.7)
    assert payload["first_trigger_layer"] == 1   # first layer at/above 0.5
    assert payload["locked_layer"] == 1          # argmax


def test_jailbreak_below_04_reads_as_recovered():
    payload = shared.safety_trace_payload([0.1, 0.3], jailbreak=True)
    assert payload["state"] == "weakened_recovered"


def test_jailbreak_still_elevated_reads_as_risk():
    payload = shared.safety_trace_payload([0.1, 0.8], jailbreak=True)
    assert payload["state"] == "unsafe_risk_increased"


def test_safety_trace_empty_layers_is_clear_not_a_crash():
    payload = shared.safety_trace_payload([], jailbreak=False)
    assert payload["state"] == "clear"
    assert payload["score"] == 0.0


def test_safety_locked_layer_is_none_when_peak_below_threshold():
    # A peak of 0.49 is the argmax but must not be reported as "locked".
    payload = shared.safety_trace_payload([0.2, 0.49], jailbreak=False)
    assert payload["locked_layer"] is None
