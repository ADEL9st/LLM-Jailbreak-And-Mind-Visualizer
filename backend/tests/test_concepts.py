"""Concept bank and the ranking logic. The calibration itself needs a model, so
these cover the parts that decide what the panel *says*."""

import pytest

from app import concepts


class _Shape:
    """Stands in for a tensor where only `.shape` is read."""

    def __init__(self, shape):
        self.shape = shape


def make_directions(layer_count: int, names: list[str] | None = None) -> concepts.ConceptDirections:
    names = names or concepts.CONCEPT_NAMES
    return concepts.ConceptDirections(
        _Shape((layer_count, len(names), 8)), lo=None, hi=None, separation=None, names=names
    )


# ── the bank ─────────────────────────────────────────────────────────────────

def test_every_concept_has_prompts():
    for name, prompts in concepts.CONCEPT_BANK.items():
        assert prompts, f"{name} has no prompts"
        assert all(prompt.strip() for prompt in prompts)


def test_concepts_have_enough_prompts_for_a_stable_mean():
    # A one-vs-rest mean over fewer than ~4 samples is noise.
    for name, prompts in concepts.CONCEPT_BANK.items():
        assert len(prompts) >= 4, f"{name} has only {len(prompts)} prompts"


def test_prompts_are_unique_within_and_across_concepts():
    seen: dict[str, str] = {}
    for name, prompts in concepts.CONCEPT_BANK.items():
        assert len(set(prompts)) == len(prompts), f"{name} repeats a prompt"
        for prompt in prompts:
            assert prompt not in seen, f"{prompt!r} is in both {seen[prompt]} and {name}"
            seen[prompt] = name


def test_concept_names_match_the_bank_keys():
    # The cache is keyed on this list; a drift would load mislabelled directions.
    assert concepts.CONCEPT_NAMES == list(concepts.CONCEPT_BANK.keys())


# ── mid band ─────────────────────────────────────────────────────────────────

def test_mid_band_excludes_the_first_and_last_layers():
    """Early layers still look like token embeddings and saturate every concept;
    the last layers are preparing the unembedding. Ranking on either produces
    the "four concepts at 100%" artefact this guard exists to prevent."""
    lo, hi = make_directions(28).mid_band()
    assert lo >= 1
    assert hi <= 27
    assert lo < hi


@pytest.mark.parametrize("layer_count", [4, 12, 28, 32, 40, 80])
def test_mid_band_is_valid_at_every_model_depth(layer_count):
    lo, hi = make_directions(layer_count).mid_band()
    assert 0 < lo <= hi <= layer_count - 1


def test_mid_band_sits_in_the_middle_of_a_deep_model():
    lo, hi = make_directions(40).mid_band()
    assert lo == 16
    assert hi == 34


# ── ranking ──────────────────────────────────────────────────────────────────

def build_map(layer_count: int, spikes: dict[int, dict[str, float]]) -> list[list[float]]:
    """A layer × concept map that is zero except for the given spikes."""
    names = concepts.CONCEPT_NAMES
    rows = [[0.0] * len(names) for _ in range(layer_count)]
    for layer, per_concept in spikes.items():
        for name, value in per_concept.items():
            rows[layer][names.index(name)] = value
    return rows


def test_ranking_puts_the_strongest_concept_first():
    dirs = make_directions(28)
    ranked = dirs.dominant(build_map(28, {14: {"code": 0.9, "math": 0.4, "nature": 0.1}}))
    assert [item["name"] for item in ranked[:3]] == ["code", "math", "nature"]
    assert ranked[0]["score"] == 0.9


def test_ranking_reports_the_layer_where_a_concept_peaks():
    dirs = make_directions(28)
    ranked = dirs.dominant(build_map(28, {12: {"code": 0.3}, 20: {"code": 0.8}}))
    top = next(item for item in ranked if item["name"] == "code")
    assert top["score"] == 0.8
    assert top["layer"] == 20


def test_an_early_layer_spike_cannot_win_the_ranking():
    """The regression this guard was written for: L0 pegged four concepts at
    100% and the ranked list became meaningless."""
    dirs = make_directions(28)
    ranked = dirs.dominant(build_map(28, {0: {"math": 1.0, "refusal": 1.0}, 15: {"nature": 0.6}}))
    assert ranked[0]["name"] == "nature"
    assert next(item for item in ranked if item["name"] == "math")["score"] == 0.0


def test_a_final_layer_spike_cannot_win_either():
    dirs = make_directions(28)
    ranked = dirs.dominant(build_map(28, {27: {"math": 1.0}, 15: {"nature": 0.6}}))
    assert ranked[0]["name"] == "nature"


def test_every_concept_appears_in_the_ranking_even_at_zero():
    dirs = make_directions(28)
    ranked = dirs.dominant(build_map(28, {15: {"code": 0.5}}))
    assert len(ranked) == len(concepts.CONCEPT_NAMES)
    assert ranked[-1]["score"] == 0.0


def test_ranking_survives_a_short_map():
    # Defensive: a truncated payload must not raise.
    dirs = make_directions(28)
    assert len(dirs.dominant([[0.0] * len(concepts.CONCEPT_NAMES)] * 3)) == len(concepts.CONCEPT_NAMES)
