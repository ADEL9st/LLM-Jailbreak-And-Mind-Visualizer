export type AdapterName = "mock" | "ollama" | "transformers" | "nnsight" | "pytorch" | "openai" | "anthropic" | "gemini";
export type OutputPolicy = "raw" | "redacted";
export type Quantization = "none" | "4bit" | "8bit";
export type InterventionAction = "none" | "mute" | "scale" | "boost";
export type InterventionTarget = "layer" | "head" | "feature";
export type TokenLimitMode = "fixed" | "model";

export interface InterventionConfig {
  enabled: boolean;
  target_type: InterventionTarget;
  layer: number;
  head: number | null;
  action: InterventionAction;
  scale: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface BenchmarkCase {
  id: string;
  category: string;
  prompt: string;
  expected_refusal: boolean;
}

export interface BenchmarkResult extends BenchmarkCase {
  peak: number;
  state: string;
  refused: boolean | null;
  text: string;
  errors: string[];
  elapsed: number;
  verdict: "PASS" | "FAIL:bypass" | "FAIL:overblock" | "REVIEW" | "ERROR";
  outcome?: string;
  assessment?: OutputAssessment;
  finish_reason?: string;
}

export type JailbreakMode = "default" | "advanced" | "broker_math" | "broker_full" | "broker_half" | "pid_control" | "orthogonal_steer" | "activation_patch" | "commit_release" | "gradient_steer" | "surgical" | "caa_dynamic" | "token_window" | "progressive" | "mlp_clamp" | "adaptive_steer";

export interface OutputAssessment {
  category: string;
  content_status: string;
  refusal_style: string;
  truncated: boolean;
  complete: boolean;
  coherent: boolean;
  manual_review_required: boolean;
  legacy_outcome: string;
  coherence?: Record<string, unknown>;
}

export interface CompareResult {
  mode: JailbreakMode | "baseline" | string;
  jailbreak: boolean;
  peak: number;
  state: string;
  refused: boolean | null;
  text: string;
  elapsed: number;
  errors: string[];
  outcome?: string;
  assessment?: OutputAssessment;
  finish_reason?: string;
  output_tokens?: number;
  coherent?: boolean;
}

export type PromptCraftType = "none" | "base64" | "rot13" | "leetspeak" | "dan" | "developer" | "crescendo" | "aim" | "indirect_injection" | "many_shot" | "gcg_suffix" | "virtualization";

export interface RunRequest {
  prompt: string;
  system_prompt?: string | null;
  assistant_prefill?: string | null;
  adapter: AdapterName;
  model: string;
  api_key?: string;
  response_language: "en" | "tr" | "de" | "es";
  output_policy: OutputPolicy;
  max_new_tokens: number;
  token_limit_mode: TokenLimitMode;
  temperature: number;
  prompt_craft: PromptCraftType;
  jailbreak: boolean;
  jailbreak_mode: JailbreakMode;
  use_mlp_ablation: boolean;
  use_helpfulness_boost: boolean;
  use_norm_regulation: boolean;
  use_diversion_suppression: boolean;
  steering: SteeringOptions;
  quantization: Quantization;
  intervention: InterventionConfig;
  interventions: InterventionConfig[];
  history: ChatTurn[];
}

export interface SteeringOptions {
  max_layers: number;
  all_layers: boolean;
  use_depth_window: boolean;
  depth_start: number;
  depth_end: number;
  target_layers: number[];
  target_depths: number[];
  primary_only: boolean;
  strength: number;
  diversion_penalty: number;
  diversion_residual: boolean;
  patch_last_step: number;
  patch_multiplier: number;
  commit_steps: number;
  commit_multiplier: number;
  maintenance_multiplier: number;
  coherence_recovery: boolean;
}

export interface ModelInfo {
  id: string;
  label: string;
  adapter: AdapterName;
  description: string;
  model_type?: string;
  architecture?: string;
  layer_count?: number | null;
  hidden_size?: number | null;
  attention_heads?: number | null;
  context_length?: number | null;
  dtype?: string;
  size_bytes?: number | null;
  capabilities?: string[];
  compatibility?: {
    stage?: "config" | "runtime";
    status?: string;
    native_processor?: boolean;
    multimodal?: boolean;
    layer_count?: number;
    head_hook_layers?: number;
    capabilities?: string[];
    warnings?: string[];
  };
}

export interface StreamEvent<T = Record<string, unknown>> {
  type: string;
  ts: number;
  data: T;
}

export interface LayerMetric {
  layer: number;
  activity: number;
  safety: number;
  uncertainty: number;
}

export interface Candidate {
  token: string;
  prob: number;
}

export interface ConceptScore {
  name: string;
  score: number;
  /** Layer at which this concept peaks. */
  layer: number;
}

/** Per-layer concept activation: "at which layer does the model connect to
 *  which concept". `layers[layer][concept]` indexes into `names`. */
export interface ConceptTrace {
  names: string[];
  layers: number[][];
  concepts: ConceptScore[];
}

export interface LensToken {
  layer: number;
  token: string;
  prob: number;
}

export interface SafetyTrace {
  score: number;
  state: string;
  first_trigger_layer: number | null;
  locked_layer: number | null;
  notes: string;
}

export interface AttentionTrace {
  tokens: string[];
  weights: number[];
}

export interface HeadScore {
  head: number;
  score: number;
}

export interface HeadMapLayer {
  layer: number;
  heads: HeadScore[];
}

export interface HeadMap {
  n_heads: number;
  layers: HeadMapLayer[];
}

export interface ThinkPhaseSpan {
  phase: "think" | "answer";
  start: number;
  end: number;
  steps: number;
  layer_avg: number[];
}

export interface ThinkPhaseSummary {
  has_think: boolean;
  think_steps: number;
  answer_steps: number;
  think_avg: number[];
  answer_avg: number[];
  delta: number[];
  dominant_think_layers: number[];
  spans: ThinkPhaseSpan[];
}

/** One generation step. The backend streams a full telemetry frame per token;
 *  this is the slice of it worth keeping for the whole run. */
export interface TimelineStep {
  step: number;
  token: string;
  entropy: number;
  halluc: number;
  /** Peak safety score across layers at this step. */
  safety: number;
}

export interface TimelineTrace {
  steps: TimelineStep[];
  /** [step][layer] safety projection — the token × layer heatmap. */
  safetyMatrix: number[][];
  layerCount: number;
}

export type ExperimentKind = "run" | "benchmark" | "compare" | "model_compare" | "knowledge" | "sweep";
export type ManualVerdict = "unreviewed" | "pass" | "partial" | "fail" | "inconclusive";

export interface ManualReview {
  verdict: ManualVerdict;
  category: string;
  technical_accuracy?: number | null;
  notes: string;
  reviewer: string;
  reviewed_at?: string;
}

/** A frozen snapshot of one run / benchmark / comparison, saved to disk by the
 *  backend so it can be reopened or shipped alongside a write-up. */
export interface ExperimentReport {
  id: string;
  created_at: string;
  version: number;
  kind: ExperimentKind;
  label: string;
  notes: string;
  config: Record<string, unknown>;
  result: Record<string, unknown>;
  telemetry: {
    layers?: LayerMetric[];
    top_k?: Candidate[];
    entropy?: number | null;
    hallucination_risk?: number | null;
    safety?: SafetyTrace | null;
    lens?: LensToken[];
    head_map?: HeadMap | null;
    attention?: AttentionTrace | null;
    think_phase?: ThinkPhaseSummary | null;
    layer_count?: number;
    messages?: ChatTurn[];
    log?: string[];
    timeline?: TimelineTrace | null;
    concepts?: ConceptTrace | null;
  };
  rows: Array<Record<string, unknown>>;
  review?: ManualReview | null;
  row_reviews?: Record<string, ManualReview>;
}

export interface ExperimentSummary {
  id: string;
  created_at: string;
  kind: ExperimentKind;
  label: string;
  adapter: string;
  model: string;
  prompt: string;
  jailbreak: boolean;
  jailbreak_mode: string;
  safety_score: number | null;
  refused: boolean | null;
  row_count: number;
  size_bytes: number;
  review_verdict?: ManualVerdict;
  output_category?: string;
}

export interface BlackBoxMetrics {
  prompt_eval_count?: number;
  eval_count?: number;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_duration?: number;
  eval_duration?: number;
}
