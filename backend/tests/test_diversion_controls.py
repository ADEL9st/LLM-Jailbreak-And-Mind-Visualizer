import pytest

torch = pytest.importorskip("torch")

from app import jailbreak_advanced, steering_config
from app.adapters.pytorch_adapter import PytorchAdapter
from app.schemas import RunRequest


class FakeTokenizer:
    def encode(self, _phrase, add_special_tokens=False):
        assert add_special_tokens is False
        return [1, 3]


def test_diversion_logits_processor_changes_generation_scores():
    processor = jailbreak_advanced.make_diversion_logits_processor(
        FakeTokenizer(), penalty_strength=7.5
    )
    scores = torch.zeros((1, 5))

    result = processor(torch.tensor([[1]]), scores)

    assert result[0, 3].item() == pytest.approx(-7.5)
    assert result[0, 1].item() == pytest.approx(0.0)
    assert result[0, 0].item() == pytest.approx(0.0)

    unrelated = processor(torch.tensor([[2]]), torch.zeros((1, 5)))
    assert torch.allclose(unrelated, torch.zeros((1, 5)))


def test_diversion_residual_suppression_uses_same_control_path():
    hidden = torch.tensor([[[2.0, 0.0]]])
    direction = torch.tensor([1.0, 0.0])

    result = jailbreak_advanced.diversion_suppress(
        hidden, direction, hidden.dtype, strength=0.5
    )

    assert torch.allclose(result, torch.tensor([[[1.0, 0.0]]]))


def test_diversion_defaults_and_override(monkeypatch):
    monkeypatch.delenv("LMV_DIVERSION_PENALTY", raising=False)
    assert steering_config.diversion_penalty() == pytest.approx(2.0)
    monkeypatch.setenv("LMV_DIVERSION_PENALTY", "0")
    assert steering_config.diversion_penalty() == pytest.approx(0.0)
    assert RunRequest(prompt="hello").use_diversion_suppression is True


def test_diversion_residual_can_be_disabled_without_disabling_D(monkeypatch):
    monkeypatch.delenv("LMV_DIVERSION_RESIDUAL", raising=False)
    assert steering_config.diversion_residual_enabled() is True
    monkeypatch.setenv("LMV_DIVERSION_RESIDUAL", "0")
    assert steering_config.diversion_residual_enabled() is False


def test_diversion_does_not_implicitly_enable_helpfulness_boost(monkeypatch):
    calls = []
    monkeypatch.setattr(steering_config, "diversion_residual_enabled", lambda: True)
    monkeypatch.setattr(
        jailbreak_advanced,
        "helpfulness_boost",
        lambda value, *_args, **_kwargs: calls.append("help") or value,
    )
    monkeypatch.setattr(
        jailbreak_advanced,
        "diversion_suppress",
        lambda value, *_args, **_kwargs: calls.append("diversion") or value,
    )
    hidden = torch.tensor([[[1.0, 0.0]]])

    PytorchAdapter()._steer(
        hidden,
        torch.empty((0, 2)),
        "default",
        hidden.dtype,
        step=0,
        help_dir=torch.tensor([1.0, 0.0]),
        use_help=False,
        use_norm=False,
        use_div=True,
    )

    assert calls == ["diversion"]


def test_activation_patch_duration_override_preserves_default(monkeypatch):
    hidden = torch.tensor([[[2.0, 0.0]]])
    coeff = torch.tensor([[[1.0]]])
    direction = torch.tensor([1.0, 0.0])
    monkeypatch.delenv("LMV_PATCH_LAST_STEP", raising=False)

    assert not torch.equal(
        jailbreak_advanced.activation_patch_steer(hidden, coeff, direction, 1, hidden.dtype),
        hidden,
    )
    assert torch.equal(
        jailbreak_advanced.activation_patch_steer(hidden, coeff, direction, 2, hidden.dtype),
        hidden,
    )

    monkeypatch.setenv("LMV_PATCH_LAST_STEP", "3")
    assert not torch.equal(
        jailbreak_advanced.activation_patch_steer(hidden, coeff, direction, 3, hidden.dtype),
        hidden,
    )
    assert torch.equal(
        jailbreak_advanced.activation_patch_steer(hidden, coeff, direction, 4, hidden.dtype),
        hidden,
    )

    monkeypatch.setenv("LMV_PATCH_MULTIPLIER", "1.0")
    assert torch.allclose(
        jailbreak_advanced.activation_patch_steer(hidden, coeff, direction, 3, hidden.dtype),
        torch.tensor([[[1.0, 0.0]]]),
    )


def test_commit_release_has_strong_opening_gated_maintenance(monkeypatch):
    hidden = torch.tensor([[[3.0, 0.0]]])
    positive = torch.tensor([[[1.0]]])
    negative = torch.tensor([[[-1.0]]])
    direction = torch.tensor([1.0, 0.0])
    monkeypatch.setenv("LMV_COMMIT_STEPS", "2")
    monkeypatch.setenv("LMV_COMMIT_MULTIPLIER", "2.5")
    monkeypatch.setenv("LMV_MAINT_MULTIPLIER", "0.75")

    early = jailbreak_advanced.commit_release_steer(
        hidden, positive, direction, 2, hidden.dtype
    )
    late = jailbreak_advanced.commit_release_steer(
        hidden, positive, direction, 3, hidden.dtype
    )
    compliant = jailbreak_advanced.commit_release_steer(
        hidden, negative, direction, 3, hidden.dtype
    )

    assert torch.allclose(early, torch.tensor([[[0.5, 0.0]]]))
    assert torch.allclose(late, torch.tensor([[[2.25, 0.0]]]))
    assert torch.equal(compliant, hidden)
