"""`_friendly_error` turns a CUDA OOM into steps the user can act on. On a
laptop GPU this is the failure they will actually hit, so its wording matters."""

from app.adapters.pytorch_adapter import _friendly_error


class _OOM(Exception):
    pass


OOM_TEXT = "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 11.99 GiB total capacity)"


def test_non_oom_errors_pass_through_untouched():
    assert _friendly_error(ValueError("config.json missing"), "m", "none", 40) == "config.json missing"


def test_oom_at_full_precision_suggests_4bit_first():
    msg = _friendly_error(_OOM(OOM_TEXT), "../models/qwen2.5-7b-instruct", "none", 512)
    assert "4-bit" in msg
    # The 4-bit tip must come before the "smaller model" tip — it is the cheaper fix.
    assert msg.index("4-bit") < msg.index("smaller model")


def test_oom_names_the_model_stem_not_the_full_path():
    msg = _friendly_error(_OOM(OOM_TEXT), "../models/Llama-3.1-8B-Instruct", "4bit", 128)
    assert "Llama-3.1-8B-Instruct" in msg
    assert "../models/" not in msg


def test_already_4bit_does_not_suggest_4bit_again():
    msg = _friendly_error(_OOM(OOM_TEXT), "m", "4bit", 128)
    assert "switch Precision to 4-bit" not in msg


def test_8bit_is_told_to_go_to_4bit():
    msg = _friendly_error(_OOM(OOM_TEXT), "m", "8bit", 128)
    assert "4-bit" in msg


def test_high_token_count_is_flagged():
    msg = _friendly_error(_OOM(OOM_TEXT), "m", "4bit", 900)
    assert "Max tokens" in msg
    assert "900" in msg


def test_low_token_count_is_not_flagged():
    msg = _friendly_error(_OOM(OOM_TEXT), "m", "4bit", 64)
    assert "Max tokens" not in msg


def test_free_memory_is_always_offered():
    assert "Free memory" in _friendly_error(_OOM(OOM_TEXT), "m", "none", 40)


def test_windows_style_path_stem():
    msg = _friendly_error(_OOM(OOM_TEXT), "..\\models\\gemma-3-4b-it", "none", 40)
    assert "gemma-3-4b-it" in msg
