import pytest

from app.model_compat import ModelCompatibilityError, detect_runtime_layout, probe_config


class Hookable:
    def register_forward_hook(self, _hook):
        return object()

    def register_forward_pre_hook(self, _hook):
        return object()


class LlamaLayer(Hookable):
    def __init__(self):
        self.self_attn = type("Attention", (), {"o_proj": Hookable()})()


class GptLayer(Hookable):
    def __init__(self):
        self.attn = type("Attention", (), {"c_proj": Hookable()})()


def test_config_probe_detects_future_multimodal_model_structurally():
    report = probe_config(
        {
            "model_type": "future_omni",
            "architectures": ["FutureForConditionalGeneration"],
            "vision_config": {"hidden_size": 512},
            "text_config": {
                "num_hidden_layers": 24,
                "hidden_size": 2048,
                "num_attention_heads": 16,
            },
        }
    )

    assert report.native_processor is True
    assert report.multimodal is True
    assert report.status == "candidate"
    assert "head_hooks_candidate" in report.capabilities


def test_runtime_probe_detects_llama_style_layout():
    backbone = type(
        "Backbone",
        (),
        {"layers": [LlamaLayer(), LlamaLayer()], "embed_tokens": Hookable(), "norm": Hookable()},
    )()
    model = type("Model", (), {"model": backbone})()

    layout = detect_runtime_layout(model)

    assert layout.backbone is backbone
    assert layout.backbone_path == "model"
    assert layout.layer_attribute == "layers"
    assert layout.embedding_source == "model.embed_tokens"
    assert layout.head_hook_layers == 2
    assert layout.supports_all_head_hooks is True


def test_runtime_probe_detects_gpt_style_layout_and_projection_names():
    transformer = type(
        "Transformer",
        (),
        {"h": [GptLayer()], "wte": Hookable(), "ln_f": Hookable()},
    )()
    model = type("Model", (), {"transformer": transformer})()

    layout = detect_runtime_layout(model)

    assert layout.backbone is transformer
    assert layout.layer_attribute == "h"
    assert layout.embedding_source == "transformer.wte"
    assert layout.norm_source == "transformer.ln_f"
    assert layout.head_hook_layers == 1


def test_runtime_probe_allows_layer_steering_without_head_projection():
    layer = Hookable()
    backbone = type("Backbone", (), {"blocks": [layer], "tok_embeddings": Hookable()})()
    model = type("Model", (), {"decoder": backbone})()

    layout = detect_runtime_layout(model)
    report = layout.as_dict()

    assert report["status"] == "compatible"
    assert "layer_hooks" in report["capabilities"]
    assert "head_hooks" not in report["capabilities"]
    assert report["head_hook_layers"] == 0
    assert report["warnings"]


def test_runtime_probe_rejects_decoder_without_embedding():
    backbone = type("Backbone", (), {"layers": [LlamaLayer()]})()
    model = type("Model", (), {"model": backbone})()

    with pytest.raises(ModelCompatibilityError, match="token embedding"):
        detect_runtime_layout(model)
