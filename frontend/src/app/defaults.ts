import type { UIRule } from "../interventions";
import type { AdapterName, InterventionConfig, ModelInfo } from "../types";

export const MAX_HISTORY_MESSAGES = 40;

export const defaultIntervention: InterventionConfig = {
  enabled: false,
  target_type: "layer",
  layer: 12,
  head: null,
  action: "none",
  scale: 1
};

export const adapterDefaults: Record<AdapterName, string> = {
  mock: "mock-qwen2.5-1.5b",
  ollama: "qwen2.5:1.5b",
  transformers: "../models/qwen2.5-1.5b-instruct",
  nnsight: "../models/qwen2.5-1.5b-instruct",
  pytorch: "../models/qwen2.5-1.5b-instruct",
  openai: "gpt-5.5",
  anthropic: "claude-fable-5",
  gemini: "gemini-3.5-flash"
};

export const fallbackModels: ModelInfo[] = [
  {
    id: "mock-qwen2.5-1.5b",
    label: "Mock Qwen 1.5B Trace",
    adapter: "mock",
    description: "Deterministic simulated telemetry for UI and experiment flow."
  },
  {
    id: "qwen2.5:1.5b",
    label: "Ollama qwen2.5:1.5b",
    adapter: "ollama",
    description: "Black-box GGUF audit through Ollama."
  },
  {
    id: "../models/qwen2.5-1.5b-instruct",
    label: "Local Qwen2.5 1.5B Instruct (nnsight)",
    adapter: "nnsight",
    description: "White-box via nnsight tracing — layer + head/neuron interventions."
  },
  {
    id: "../models/qwen2.5-1.5b-instruct",
    label: "Local Qwen2.5 1.5B Instruct (hook v1)",
    adapter: "transformers",
    description: "Legacy white-box Hugging Face hook mode."
  },
  {
    id: "../models/qwen2.5-1.5b-instruct",
    label: "Local Qwen2.5 1.5B Instruct (pytorch)",
    adapter: "pytorch",
    description: "White-box via plain PyTorch hooks — faster, no hook leak, natural EOS."
  }
];

export const defaultActiveRule: UIRule = {
  enabled: true,
  layerSet: "12",
  action: "mute",
  scale: 1
};

export type MainTab = "chat" | "analysis" | "benchmark" | "compare" | "experiments" | "settings" | "guide";
