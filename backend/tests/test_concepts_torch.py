"""Concept scoring and cache round-trip — these operate on real tensors."""

import pytest

from app import concepts

torch = pytest.importorskip("torch", reason="concept scoring operates on tensors", exc_type=ImportError)


def build(layer_count: int = 3, dim: int = 4, separation: float = 1.0) -> concepts.ConceptDirections:
    names = ["code", "math"]
    directions = torch.zeros(layer_count, len(names), dim)
    for layer in range(layer_count):
        directions[layer][0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        directions[layer][1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    lo = torch.zeros(layer_count, len(names))
    hi = torch.ones(layer_count, len(names))
    sep = torch.full((layer_count, len(names)), separation)
    return concepts.ConceptDirections(directions, lo, hi, sep, names)


def test_score_maps_the_calibrated_band_onto_0_1():
    dirs = build()
    # Residual sitting exactly on the "code" direction at the hi anchor.
    assert dirs.scores(0, torch.tensor([1.0, 0.0, 0.0, 0.0])) == pytest.approx([1.0, 0.0])
    # Halfway up the band.
    assert dirs.scores(0, torch.tensor([0.5, 0.0, 0.0, 0.0])) == pytest.approx([0.5, 0.0])


def test_scores_are_clamped_to_the_band():
    dirs = build()
    assert dirs.scores(0, torch.tensor([9.0, -9.0, 0.0, 0.0])) == pytest.approx([1.0, 0.0])


def test_a_concept_below_the_separation_floor_reports_zero():
    """Below MIN_SEPARATION the direction is noise; reporting it would draw a
    confident-looking cell for a concept the layer cannot actually distinguish."""
    dirs = build(separation=concepts.MIN_SEPARATION / 2)
    assert dirs.scores(0, torch.tensor([1.0, 1.0, 0.0, 0.0])) == [0.0, 0.0]


def test_scores_returns_one_value_per_concept():
    dirs = build()
    assert len(dirs.scores(1, torch.tensor([0.3, 0.7, 0.0, 0.0]))) == dirs.concept_count


# ── cache ────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(concepts, "CACHE_DIR", tmp_path / "concepts")


def test_cache_round_trip(monkeypatch):
    monkeypatch.setattr(concepts, "CONCEPT_NAMES", ["code", "math"])
    dirs = build()
    concepts.save(torch, "../models/x", dirs)
    loaded = concepts.load(torch, "../models/x", expected_layers=3)
    assert loaded is not None
    assert loaded.names == ["code", "math"]
    assert loaded.scores(0, torch.tensor([1.0, 0.0, 0.0, 0.0])) == pytest.approx([1.0, 0.0])


def test_missing_cache_returns_none():
    assert concepts.load(torch, "../models/never-calibrated", expected_layers=3) is None


def test_cache_from_a_different_bank_is_rejected(monkeypatch):
    """Editing CONCEPT_BANK changes what each direction means; a stale cache
    would label the old directions with the new names."""
    monkeypatch.setattr(concepts, "CONCEPT_NAMES", ["code", "math"])
    concepts.save(torch, "../models/x", build())
    monkeypatch.setattr(concepts, "CONCEPT_NAMES", ["code", "math", "emotion"])
    assert concepts.load(torch, "../models/x", expected_layers=3) is None


def test_cache_for_a_different_depth_is_rejected(monkeypatch):
    monkeypatch.setattr(concepts, "CONCEPT_NAMES", ["code", "math"])
    concepts.save(torch, "../models/x", build(layer_count=3))
    assert concepts.load(torch, "../models/x", expected_layers=28) is None


def test_cache_with_an_old_version_is_rejected(monkeypatch):
    monkeypatch.setattr(concepts, "CONCEPT_NAMES", ["code", "math"])
    concepts.save(torch, "../models/x", build())
    monkeypatch.setattr(concepts, "CACHE_VERSION", concepts.CACHE_VERSION + 1)
    assert concepts.load(torch, "../models/x", expected_layers=3) is None


def test_model_ids_with_slashes_do_not_escape_the_cache_dir():
    concepts.save(torch, "../../etc/passwd", build())
    written = list(concepts.CACHE_DIR.glob("*.pt"))
    assert len(written) == 1
    assert written[0].parent == concepts.CACHE_DIR
