"""`raw_activities` operates on tensors, so it needs PyTorch. Kept in its own
module: a module-level importorskip aborts the whole file, and the rest of the
shared-module tests must still run in a torch-less environment (CI)."""

import pytest

from app.adapters import shared

# exc_type is pytest 9.1's default; setting it now keeps the skip working and
# silences the deprecation warning.
torch = pytest.importorskip("torch", reason="raw_activities operates on tensors", exc_type=ImportError)



@pytest.mark.torch
def test_raw_activities_is_zero_when_no_layer_changes_the_residual():
    embed = torch.ones(8)
    layers = [torch.ones(8), torch.ones(8)]
    assert shared.raw_activities(layers, embed) == pytest.approx([0.0, 0.0])


@pytest.mark.torch
def test_raw_activities_is_scale_invariant():
    """Activity is delta/(‖before‖+‖after‖) — a *relative* measure. Scaling the
    whole residual stream must not change it, otherwise layer activity would
    track the model's absolute activation magnitude instead of how much each
    layer rewrites the stream."""
    small = shared.raw_activities([torch.ones(8) * 2], torch.ones(8))
    large = shared.raw_activities([torch.ones(8) * 20], torch.ones(8) * 10)
    assert small[0] == pytest.approx(large[0])


@pytest.mark.torch
def test_raw_activities_shrinks_as_the_residual_it_edits_grows():
    # The same absolute delta counts for less on top of a larger residual.
    on_small = shared.raw_activities([torch.full((8,), 1.1)], torch.ones(8))[0]
    on_large = shared.raw_activities([torch.full((8,), 10.1)], torch.full((8,), 10.0))[0]
    assert on_large < on_small


@pytest.mark.torch
def test_raw_activities_measures_each_layer_against_the_previous_one():
    # Layer 0 moves the residual, layer 1 leaves it alone — so only the first
    # entry should be non-zero.
    embed = torch.zeros(8)
    out = shared.raw_activities([torch.ones(8), torch.ones(8)], embed)
    assert out[0] > 0.0
    assert out[1] == pytest.approx(0.0)
