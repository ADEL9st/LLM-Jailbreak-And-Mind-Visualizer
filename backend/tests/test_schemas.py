"""Request validation. These bounds are the only thing between a hostile or
buggy client and a model load, so they are worth pinning."""

import pytest
from pydantic import ValidationError

from app.schemas import InterventionConfig, RunRequest


def make_request(**overrides) -> RunRequest:
    return RunRequest(**{"prompt": "hello", **overrides})


def test_defaults_are_the_safe_ones():
    request = make_request()
    assert request.adapter == "mock"
    assert request.jailbreak is False
    assert request.output_policy == "raw"
    assert request.quantization == "none"
    assert request.active_interventions() == []


@pytest.mark.parametrize("field,value", [
    ("max_new_tokens", 0),
    ("max_new_tokens", 65537),
    ("token_limit_mode", "infinite"),
    ("temperature", -0.1),
    ("temperature", 2.1),
    ("adapter", "not-an-adapter"),
    ("quantization", "16bit"),
    ("output_policy", "hidden"),
    ("jailbreak_mode", "nonexistent"),
    ("prompt_craft", "nonexistent"),
])
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        make_request(**{field: value})


def test_max_new_tokens_accepts_extended_limit():
    assert make_request(max_new_tokens=65536).max_new_tokens == 65536
    assert make_request(token_limit_mode="model").token_limit_mode == "model"


def test_empty_prompt_is_rejected():
    with pytest.raises(ValidationError):
        RunRequest(prompt="")


def test_oversized_prompt_is_rejected():
    with pytest.raises(ValidationError):
        RunRequest(prompt="x" * 16001)


def test_system_prompt_is_optional_and_capped():
    assert make_request().system_prompt is None
    assert make_request(system_prompt="policy").system_prompt == "policy"
    with pytest.raises(ValidationError):
        make_request(system_prompt="x" * 16001)


def test_assistant_prefill_is_optional_and_capped():
    assert make_request().assistant_prefill is None
    assert make_request(assistant_prefill="The answer begins").assistant_prefill == "The answer begins"
    with pytest.raises(ValidationError):
        make_request(assistant_prefill="x" * 8001)


def test_request_scoped_steering_validates_targets_and_depth_window():
    request = make_request(steering={"target_layers": [3, 7], "strength": 0.65})
    assert request.steering.target_layers == [3, 7]
    assert request.steering.strength == 0.65
    with pytest.raises(ValidationError):
        make_request(steering={"target_layers": [3], "target_depths": [0.5]})
    with pytest.raises(ValidationError):
        make_request(steering={"use_depth_window": True, "depth_start": 0.8, "depth_end": 0.4})


def test_history_is_capped():
    turns = [{"role": "user", "content": "hi"} for _ in range(65)]
    with pytest.raises(ValidationError):
        make_request(history=turns)


def test_intervention_list_is_capped():
    rules = [{"enabled": True, "action": "mute", "layer": 0} for _ in range(129)]
    with pytest.raises(ValidationError):
        make_request(interventions=rules)


def test_negative_layer_is_rejected():
    with pytest.raises(ValidationError):
        InterventionConfig(layer=-1)


def test_scale_bounds():
    InterventionConfig(scale=5.0)
    InterventionConfig(scale=-5.0)
    with pytest.raises(ValidationError):
        InterventionConfig(scale=5.1)


# ── active_interventions() ───────────────────────────────────────────────────

def test_active_interventions_filters_disabled_and_noop_rules():
    request = make_request(interventions=[
        {"enabled": True, "action": "mute", "layer": 1},
        {"enabled": False, "action": "mute", "layer": 2},   # disabled
        {"enabled": True, "action": "none", "layer": 3},    # no-op
    ])
    active = request.active_interventions()
    assert [rule.layer for rule in active] == [1]


def test_singular_intervention_is_used_only_when_the_list_is_empty():
    request = make_request(intervention={"enabled": True, "action": "mute", "layer": 9})
    assert [rule.layer for rule in request.active_interventions()] == [9]


def test_list_takes_precedence_over_the_singular_field():
    request = make_request(
        intervention={"enabled": True, "action": "mute", "layer": 9},
        interventions=[{"enabled": True, "action": "mute", "layer": 1}],
    )
    assert [rule.layer for rule in request.active_interventions()] == [1]


def test_disabled_singular_intervention_yields_nothing():
    request = make_request(intervention={"enabled": False, "action": "mute", "layer": 9})
    assert request.active_interventions() == []
