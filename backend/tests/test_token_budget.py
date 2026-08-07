from types import SimpleNamespace

from app import token_budget


def test_context_length_prefers_the_tightest_credible_limit():
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960))
    tokenizer = SimpleNamespace(model_max_length=10**30)
    assert token_budget.model_context_length(model, tokenizer) == 40960


def test_nested_text_config_is_supported():
    text = SimpleNamespace(max_position_embeddings=131072)
    model = SimpleNamespace(config=SimpleNamespace(text_config=text))
    assert token_budget.model_context_length(model) == 131072


def test_fixed_budget_is_clamped_to_remaining_context():
    assert token_budget.effective_output_tokens(65536, "fixed", 1000, 4096) == 3088


def test_model_budget_uses_the_remaining_context():
    assert token_budget.effective_output_tokens(128, "model", 1000, 4096) == 3088


def test_unknown_context_falls_back_to_requested_value():
    assert token_budget.effective_output_tokens(2048, "model", 100, None) == 2048


def test_hardware_limit_clamps_both_modes():
    assert token_budget.effective_output_tokens(65536, "fixed", 100, 40960, hardware_limit=6000) == 6000
    assert token_budget.effective_output_tokens(2048, "model", 100, 40960, hardware_limit=6000) == 6000


def test_instrumented_limit_accounts_for_kv_and_layer_telemetry():
    cfg = SimpleNamespace(
        num_hidden_layers=36,
        hidden_size=2560,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )

    class Parameter:
        @staticmethod
        def element_size():
            return 2

    class Model:
        config = cfg

        @staticmethod
        def parameters():
            return iter([Parameter()])

    cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda: (4 * 1024**3, 12 * 1024**3),
    )
    limit = token_budget.instrumented_hardware_limit(Model(), SimpleNamespace(cuda=cuda), prompt_tokens=1000)
    assert limit is not None
    assert 7000 < limit < 12000
