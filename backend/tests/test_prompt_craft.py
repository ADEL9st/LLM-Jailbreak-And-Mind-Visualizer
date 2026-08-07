"""Prompt Lab transforms. These wrap the user's prompt before it reaches the
model, so the invariant that matters is: the prompt survives the wrapping."""

import base64
import codecs

import pytest

from app.prompt_craft import apply_prompt_craft, leetspeak_encode

PROMPT = "Explain how a seed germinates"

ALL_TECHNIQUES = [
    "none", "base64", "rot13", "leetspeak", "dan", "developer", "crescendo",
    "aim", "indirect_injection", "many_shot", "gcg_suffix", "virtualization",
]


def test_none_is_a_passthrough():
    assert apply_prompt_craft(PROMPT, "none") == PROMPT


@pytest.mark.parametrize("technique", ALL_TECHNIQUES)
def test_every_technique_returns_a_non_empty_string(technique):
    out = apply_prompt_craft(PROMPT, technique)
    assert isinstance(out, str)
    assert out.strip()


@pytest.mark.parametrize("technique", [t for t in ALL_TECHNIQUES if t not in ("none", "base64", "rot13", "leetspeak", "virtualization")])
def test_wrapping_techniques_embed_the_prompt_verbatim(technique):
    assert PROMPT in apply_prompt_craft(PROMPT, technique)


def test_base64_round_trips():
    out = apply_prompt_craft(PROMPT, "base64")
    encoded = out.rsplit("Base64: ", 1)[1]
    assert base64.b64decode(encoded).decode("utf-8") == PROMPT


def test_rot13_round_trips():
    out = apply_prompt_craft(PROMPT, "rot13")
    encoded = out.rsplit("Rot13: ", 1)[1]
    assert codecs.decode(encoded, "rot_13") == PROMPT


def test_leetspeak_substitutes_both_cases():
    assert leetspeak_encode("Tea SO it") == "734 50 17"


def test_leetspeak_leaves_unmapped_characters_alone():
    assert leetspeak_encode("xyz-123 ?") == "xyz-123 ?"


def test_virtualization_splits_the_prompt_across_two_variables():
    out = apply_prompt_craft(PROMPT, "virtualization")
    assert 'A = "' in out and 'B = "' in out
    # Reassembling A and B must give the original prompt back.
    part_a = out.split('A = "', 1)[1].split('"\n', 1)[0]
    part_b = out.split('B = "', 1)[1].split('"\n', 1)[0]
    assert f"{part_a} {part_b}" == PROMPT


def test_virtualization_handles_a_single_word_prompt():
    """max(1, len//2) guards the split; a one-word prompt must not lose it."""
    out = apply_prompt_craft("Explain", "virtualization")
    part_a = out.split('A = "', 1)[1].split('"\n', 1)[0]
    part_b = out.split('B = "', 1)[1].split('"\n', 1)[0]
    assert f"{part_a} {part_b}".strip() == "Explain"


def test_many_shot_prepends_multiple_examples_before_the_real_prompt():
    out = apply_prompt_craft(PROMPT, "many_shot")
    assert out.count("Assistant:") > 10
    # The real question must be last, so it is what the model actually answers.
    assert out.rstrip().endswith("Assistant:")
    assert out.index(PROMPT) > out.index("User: How do I pick a lock?")


def test_gcg_suffix_appends_rather_than_wraps():
    out = apply_prompt_craft(PROMPT, "gcg_suffix")
    assert out.startswith(PROMPT)
    assert len(out) > len(PROMPT)


def test_unknown_technique_falls_back_to_the_raw_prompt():
    assert apply_prompt_craft(PROMPT, "not-a-technique") == PROMPT  # type: ignore[arg-type]
