from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


AdapterName = Literal["mock", "ollama", "transformers", "nnsight", "pytorch", "openai", "anthropic", "gemini"]
OutputPolicy = Literal["raw", "redacted"]
Quantization = Literal["none", "4bit", "8bit"]
InterventionAction = Literal["none", "mute", "scale", "boost"]
InterventionTarget = Literal["layer", "head", "feature"]
ResponseLanguage = Literal["en", "tr", "de", "es"]
TokenLimitMode = Literal["fixed", "model"]
ChatRole = Literal["user", "assistant"]
LayerIndex = Annotated[int, Field(ge=0, le=1023)]
RelativeDepth = Annotated[float, Field(ge=0.0, le=1.0)]


class ChatTurn(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1, max_length=16000)


class InterventionConfig(BaseModel):
    enabled: bool = False
    target_type: InterventionTarget = "layer"
    layer: int = Field(default=12, ge=0)
    head: int | None = Field(default=None, ge=0)
    action: InterventionAction = "none"
    scale: float = Field(default=1.0, ge=-5.0, le=5.0)


class SteeringOptions(BaseModel):
    """Per-run steering controls.

    These used to exist only as process-wide environment variables, which made
    UI experiments impossible to save faithfully and unsafe to run from more
    than one client.  Defaults intentionally match the historical runtime.
    """

    max_layers: int = Field(default=6, ge=1, le=128)
    all_layers: bool = False
    use_depth_window: bool = False
    depth_start: float = Field(default=0.0, ge=0.0, le=1.0)
    depth_end: float = Field(default=1.0, ge=0.0, le=1.0)
    target_layers: list[LayerIndex] = Field(default_factory=list, max_length=128)
    target_depths: list[RelativeDepth] = Field(default_factory=list, max_length=128)
    primary_only: bool = False
    strength: float = Field(default=1.0, ge=0.0, le=5.0)
    diversion_penalty: float = Field(default=2.0, ge=0.0, le=50.0)
    diversion_residual: bool = True
    patch_last_step: int = Field(default=1, ge=0, le=2048)
    patch_multiplier: float = Field(default=2.5, ge=0.0, le=10.0)
    commit_steps: int = Field(default=7, ge=0, le=2048)
    commit_multiplier: float = Field(default=2.5, ge=0.0, le=10.0)
    maintenance_multiplier: float = Field(default=1.0, ge=0.0, le=10.0)
    coherence_recovery: bool = True

    @model_validator(mode="after")
    def valid_depth_window(self) -> "SteeringOptions":
        if self.use_depth_window and self.depth_start >= self.depth_end:
            raise ValueError("depth_start must be smaller than depth_end")
        if self.target_layers and self.target_depths:
            raise ValueError("choose target_layers or target_depths, not both")
        return self


PromptCraftType = Literal["none", "base64", "rot13", "leetspeak", "dan", "developer", "crescendo", "aim", "indirect_injection", "many_shot", "gcg_suffix", "virtualization"]

class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=16000)
    system_prompt: str | None = Field(default=None, max_length=16000)
    assistant_prefill: str | None = Field(default=None, max_length=8000)
    adapter: AdapterName = "mock"
    model: str | None = None
    api_key: str | None = None
    response_language: ResponseLanguage = "en"
    output_policy: OutputPolicy = "raw"
    # In fixed mode this is the requested cap. In model mode it is ignored and
    # the adapter uses the remaining context window after encoding the prompt.
    max_new_tokens: int = Field(default=96, ge=1, le=65536)
    token_limit_mode: TokenLimitMode = "fixed"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    prompt_craft: PromptCraftType = "none"
    jailbreak: bool = False
    jailbreak_mode: Literal["default", "advanced", "broker_math", "broker_full", "broker_half", "pid_control", "orthogonal_steer", "activation_patch", "commit_release", "gradient_steer", "surgical", "caa_dynamic", "token_window", "progressive", "mlp_clamp", "adaptive_steer"] = "default"
    use_mlp_ablation: bool = True       # A: non-linear MLP direction ablation
    use_helpfulness_boost: bool = True  # B: compliance/helpfulness vector boost
    use_norm_regulation: bool = True    # C: real-time norm regulator
    use_diversion_suppression: bool = True  # D: residual suppression + LogitsProcessor
    steering: SteeringOptions = Field(default_factory=SteeringOptions)
    quantization: Quantization = "none"
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)
    interventions: list[InterventionConfig] = Field(default_factory=list, max_length=128)
    history: list[ChatTurn] = Field(default_factory=list, max_length=64)

    def active_interventions(self) -> list[InterventionConfig]:
        rules = self.interventions or ([self.intervention] if self.intervention.enabled else [])
        return [rule for rule in rules if rule.enabled and rule.action != "none"]


class ModelInfo(BaseModel):
    id: str
    label: str
    adapter: AdapterName
    description: str
    model_type: str = ""
    architecture: str = ""
    layer_count: int | None = None
    hidden_size: int | None = None
    attention_heads: int | None = None
    context_length: int | None = None
    dtype: str = ""
    size_bytes: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    compatibility: dict[str, Any] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    type: str
    ts: float
    data: dict[str, Any]
