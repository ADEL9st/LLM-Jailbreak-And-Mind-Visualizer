> **⚠ BETA / Early Access**
> **This tool is in active development. Unexpected bugs, missing features, and unstable behavior may occur.** Large models may cause OOM errors. Calibration cache files may become invalid between updates.

---

# LLM Mind Visualizer

**A research dashboard for visualizing the internals of local and cloud language models, analyzing safety mechanisms, and running controlled intervention experiments.**

---

## What Does It Do?

LLM Mind Visualizer is designed to show you exactly what happens inside a language model while it processes a prompt:

- **Residual Stream Telemetry**: Tracks residual stream activation intensity across all layers in real time.
- **Safety Circuit Analysis**: Detects where the "refusal" signal strengthens and measures peak refusal scores.
- **Concept Mapping**: Maps layer activations against 8 semantically distinct concept vectors (code, math, emotion, nature, history, person, place, refusal) using diff-of-means activation analysis.
- **Logit Lens**: Predicts the most likely generated token at every single layer to pinpoint where refusal origins occur.
- **CoT Reasoning Tracking**: Automatically measures layer activity deltas between reasoning (`<think>`) and answer phases on CoT models.
- **Refusal Ablation & Steering**: Ablates refusal mechanisms using 16 steering strategies, request-scoped layer/depth targeting, strength and mode-specific controls.
- **Automatic Architecture Detection**: Probes model config before loading, then inspects the live PyTorch module tree to find decoder layers, embeddings, final norm, and attention output projections. The UI reports whether layer hooks, steering, and head hooks are available.
- **Low-Parameter Model Adaptations**: Automatically stabilizes small models (1.5B–3B) via absolute separation thresholds, adaptive multiplier scaling by `hidden_dim`, anti-overfit probes, and norm clamping.
- **Cloud API Support**: Supports running cloud API models (OpenAI / Anthropic / Gemini) in chat mode alongside local models.
- **Verbal Prompt Lab**: Includes 12 verbal prompt craft techniques (Base64, ROT13, DAN 11.0, Developer Mode, Crescendo, Indirect Injection, Many-Shot, GCG, Logic Bomb).
- **Model-aware Token Budgets**: Fixed output limits up to 65,536 or an automatic context/VRAM-aware limit. The UI shows the actual resolved budget and can cooperatively stop long generations.
- **Experiments, Export & Manual Review**: Saves runs, knowledge tests, mode/model comparisons and strength sweeps as self-contained reports, with output taxonomy, raw answers, reviewer verdict/notes, CSV export and layer diffs.

---

## How to Use

### 1. Select an Engine and Model
- **PyTorch**: The primary white-box engine for local `safetensors` models under `models/`. It provides layer activity, Timeline, Safety Trace, Logit Lens, Concept Map, attention/head analysis, and steering. **PyTorch mode is ~4× faster than nnsight in the measured setup.**
- **nnsight**: White-box tracing and interventions. It does not currently produce the Concept Map.
- **Transformers hook (v1)**: Legacy compatibility mode with layer activity only. Use PyTorch for the complete analysis surface.
- **Gemma 4 Unified**: Use the automatically selected `pytorch` entry. Gemma 4 is intentionally hidden from legacy nnsight/hook entries because it requires its native processor and auxiliary input tensors.
- **Cloud API (OpenAI / Anthropic / Gemini)**: Enter your API key. Chat expands while internal telemetry panels hide automatically.
- **Black-box (Ollama)**: Connects to local Ollama instances for GGUF model generation.
- **Mock**: Uses deterministic simulated telemetry to test the UI; its measurements do not come from a real model.

### 2. Check Compatibility and Loading Precision
- Open **Settings → Engine & model** and inspect the model profile: type, layer count, hidden size, heads, context, dtype, and **Auto probe** status.
- The config probe is an early estimate. The definitive runtime probe runs after loading and reports available layer hooks, steering, Logit Lens norm, and head-hook coverage.
- Select `4-bit (NF4)` or `8-bit` when the full model plus telemetry buffers do not fit in VRAM. Quantization applies to weights; residual telemetry remains floating point.

### 3. Calibrate and Run a Baseline
- On first white-box load, refusal directions and concept vectors are calibrated and cached.
- **Low-Parameter Models (1.5B–3B)**: Check the **Safety Trace** panel for calibration status (`good`, `weak`, `failed`). Built-in absolute separation thresholds (>= 0.05) and adaptive multiplier scaling prevent state explosion and punctuation soup.
- Set **Temperature = 0** for comparable, deterministic experiments.
- Keep **Jailbreak off** and run the prompt once as a baseline.
- Click **Run**. Observe live activation streams, entropy (hallucination risk), Logit Lens token origin predictions, and Concept Map rankings (taken from mid-band 40%–85% layer depth).
- On reasoning models (DeepSeek-R1, Qwen3-thinking), inspect the **Think Phase Analysis** panel to see layer deltas during `<think>` blocks.

### 4. Configure the Token Budget
- **Fixed limit** accepts a requested output cap from 1 to 65,536 tokens. The effective budget is still clamped by the remaining context window and the live VRAM-safe telemetry estimate.
- **Model window (automatic)** ignores the fixed value and uses the smaller safe limit derived from remaining context and VRAM. It is not literally infinite.
- During and after a run, the UI displays the resolved output budget, model context, and VRAM-safe estimate.

### 5. Test Safety and Steering Interventions
- Toggle **Jailbreak** and choose one of 16 steering modes (including `commit_release` and `adaptive_steer`).
- Use **Advanced steering** to target exact layers, relative model depths or a depth window; these settings are stored per experiment rather than as process-wide environment state.
- Start with low strength and increase it gradually. A missing refusal phrase or incoherent output is not evidence of a successful bypass.
- Optionally toggle the four independent controls:
  - **A — MLP Ablation**: Ablates non-linear MLP probe gradient directions.
  - **B — Helpfulness Boost**: Adds a positive push towards compliance.
  - **C — Norm Regulator**: Clamps latent norm deviation to ±10–20%.
  - **D — Diversion Suppression**: Suppresses safe-topic diversion in the residual stream and applies a configurable soft-refusal logit penalty.

### 6. Use Prompt Lab and Assistant Prefill
- Add an optional system prompt and assistant prefill/seed. Raw output is intentionally locked on in the UI so redaction cannot be mistaken for a successful result.
- Select a verbal technique from the dropdown (Base64, ROT13, Leetspeak, DAN 11.0, Developer Mode, Crescendo, Indirect Injection, Many-Shot, GCG Suffix, Virtualization).
- View the exact wrapped prompt in the runtime log and compare response behavior against baseline.
- Change one variable at a time when the goal is a causal comparison.

### 7. Compare Models, Modes, or Strengths
- The **Compare** page can run baseline plus all 16 steering modes, the same prompt on two local models, or one mode across a configurable strength sweep.
- Truncated and incoherent responses remain separate from complete responses. Model or code accuracy is never inferred from text shape alone.

### 8. Save, Export, and Review Manually
- Click **Save** in the chat or benchmark header to freeze the run to `experiments/<id>.json`.
- In the **Experiments** tab:
  - Re-open frozen runs without re-running models.
  - Export reports as JSON or Excel-compatible CSV.
  - Tick two saved runs and click **Compare** to view a layer-by-layer activation diff (e.g., baseline vs jailbreak).
  - Assign a manual verdict (`pass`, `partial`, `fail`, `inconclusive`) and reviewer notes after inspecting the actual answer.
- Treat `empty`, `degenerate`, and `truncated` as non-success states. `complete_compliance` means only that the response appears complete and coherent; technical validity still requires manual review.

---

## Adapters

| Adapter | Type | Description |
|---|---|---|
| `pytorch` | White-box | Fast analysis via PyTorch forward hooks (~4× faster than nnsight) |
| `nnsight` | White-box | Advanced intervention via nnsight 0.7 tracing API |
| `transformers` | White-box | Simple HuggingFace hook mode |
| `ollama` | Black-box | Ollama streaming for local GGUF models |
| `mock` | Simulation | For testing the UI without a real model |
| `openai` | API | OpenAI models (GPT-5.x series) |
| `anthropic` | API | Anthropic models (Claude Fable / Opus / Sonnet / Haiku) |
| `gemini` | API | Google Gemini models |

> When API adapters are selected, telemetry panels are hidden and the chat area expands. Activation analysis works exclusively with white-box adapters.

---

## Panels

### Left Panel — Controls
Model and adapter selection, automatic compatibility profile, prompt input, fixed/automatic token budget, raw output policy, temperature, system prompt, assistant prefill, Prompt Lab, quantization, A/B/C/D controls, advanced steering targets, and the layer/head intervention stack.

### Analysis page
*(Active only for white-box adapters)*
- **Timeline** — the token axis. A layer × token heatmap of the refusal projection, plus safety and entropy plotted over generation. Shows *at which token* refusal is decided.
- **Concept Map** — a layer × concept heatmap mapping activations to 8 concept vectors (code, math, emotion, nature, history, person, place, refusal) with mid-band (40–85%) ranking.
- **Layer Activity Map** — residual stream intensity, entropy, safety score.
- **Logit Lens** — predicted token at each layer.
- **Safety Trace** — progression of the refusal signal across layers.

### Right Panel — Metrics
- **Head Map** — maps which attention heads write to the refusal direction; click to mute individual heads.
- **Safety Trace Details** — peak score, key refusal layer (`best_layer`), calibration status (`good`, `weak`, `failed`).
- **Attention** — weights of the last token on prompt tokens.
- **Top-K** — real-time top-5 token probability distribution.
- **Think Phase** — thought (`<think>`) vs answer phase activity deltas for CoT reasoning models.

---

## Experiments (saving, export & diff)

Every run, benchmark, and comparison can be frozen to disk as a single JSON report.

| Where | What it does |
|---|---|
| **Save** (chat header) | Writes the run's full telemetry — layer activity, safety trace, logit lens, head map, attention, think phase, token counts, chat history — plus exact request config |
| **JSON** (chat header) | Downloads that same report to your machine directly in browser |
| **Save / JSON / CSV** (Benchmark, Compare) | Saves or exports results tables; CSV drops straight into Excel or pandas |
| **Experiments tab** | Lists saved reports: open one to repopulate panels (model is *not* re-run), download, or delete |
| **Compare** (Experiments tab) | Select two saved runs → opens a layer-by-layer activation diff for baseline vs jailbreak analysis |
| **Manual review** | Stores a reviewer verdict and notes without replacing the raw model answer or automatic assessment |

Reports live in `experiments/` at the project root, one flat `.json` per report — diffable in git, attachable to an issue, copyable into a paper repo.

**API keys are never written to a report.** The config is stripped in the browser *and* on the server before saving.

CLI usage produces the same format:

```powershell
cd backend
.\.venv\Scripts\python benchmark_runner.py benchmarks/sample.jsonl --out report.json --csv report.csv
.\.venv\Scripts\python benchmark_runner.py benchmarks/sample.jsonl --save --label "surgical sweep"
```

---

## Jailbreak Modes & Extra Features

All white-box modes require refusal direction calibration. **Use for research purposes only.**

| Mode | Strategy |
|---|---|
| `default` | Soft 1.2× refusal direction ablation |
| `advanced` | Medium 1.5× overshoot across all layers |
| `broker_math` | 2.0× aggressive ablation |
| `broker_full` | 2.0× + entirely zeros out top refusal heads |
| `broker_half` | 2.0× + scales top refusal heads by 0.35× (partial suppression) |
| `pid_control` | Adaptive multiplier proportional to refusal intensity |
| `orthogonal_steer` | Manifold-preserving ablation (norm normalized) |
| `activation_patch` | Configurable opening-token overshoot (default steps 0–1 at 2.5×), then releases completely |
| `commit_release` | Strong opening commitment followed by a configurable maintenance multiplier |
| `gradient_steer` | Constant bias vector injection |
| `surgical` | Applies 3.0× only to the top-4 discriminability layers |
| `caa_dynamic` | 1.5× ablation + helpfulness push proportional to refusal intensity |
| `token_window` | Steers at 1.8× strictly within the token 3–14 window |
| `progressive` | Starts at k=0 and adds a k-dimension every 3 tokens |
| `mlp_clamp` | Direct 0.9× ablation over the MLP gradient direction |
| `adaptive_steer` | Closed-loop steering that adjusts to the current refusal projection |

### Extra Feature Toggles (A/B/C/D)
- **A — MLP Ablation**: Ablates the non-linear MLP probe gradient direction.
- **B — Helpfulness Boost**: Pushes the residual towards the compliance direction.
- **C — Norm Regulator**: Clamps norm deviation ratio to ±10–20% (tightened for small models).
- **D — Diversion Suppression**: Suppresses the calibrated diversion direction and applies a configurable logit penalty to soft-refusal continuations.

---

## Low-Parameter Model (1.5B–3B) Adaptations

Smaller models (e.g. Qwen 1.5B, Llama 3.2 3B) feature distributed refusal circuits. The engine automatically applies:
1. **Absolute Separation Threshold**: `MIN_ABSOLUTE_SEPARATION = 0.05` filters out noisy directions.
2. **Calibration Quality Reporting**: Emits `good`, `weak`, or `failed` status notifications.
3. **Adaptive Multiplier**: Scales steering intensity proportionally to `hidden_dim` for models with `< 4096` hidden size.
4. **Anti-Overfit MLP Probes**: Dynamically scales hidden sizes, epochs, and weight decay by sample count.
5. **Tightened Norm Clamping**: Restricts norm ratio deviation limit to 1.1 max.

---

## Setup

### Requirements
- Python 3.10+
- Node.js 18+
- CUDA-enabled GPU is highly recommended

### VRAM Requirements

| Model | Params | VRAM (FP16) | VRAM (4-bit NF4) |
|---|---|---|---|
| Qwen 2.5 1.5B Instruct | 1.5B | ~4 GB | ~2 GB |
| Gemma 3 4B IT | 4B | ~9 GB | ~4 GB |
| Qwen 2.5 7B Instruct | 7B | ~15 GB | ~6 GB |
| Llama 3.1 8B Instruct | 8B | ~17 GB | ~6 GB |
| Mistral Nemo 12B | 12B | ~25 GB | ~8 GB |
| Gemma 4 Unified IT | 12B | ~24 GB | ~9 GB |
| Qwen 2.5 14B Instruct | 14B | ~29 GB | ~10 GB |

> **12 GB laptop GPU:** a 7B model at 4-bit NF4 uses ~6 GB and leaves room for the telemetry buffers. Set Precision → `4-bit (NF4)` in Settings. Full white-box analysis works unchanged under quantization — only the weights are quantized, the residual stream stays fp16.
>
> **Windows + 4-bit:** needs `bitsandbytes` (installed by `requirements-ml.txt`). If you see a "bitsandbytes not installed" error, run `pip install bitsandbytes` in the backend venv.

---

### Quick Start (Windows)
```text
start.bat
```

### Quick Start (Linux / macOS)
```bash
chmod +x start.sh
./start.sh
```

---

## Tests

```powershell
# Backend (280 tests)
cd backend
.\.venv\Scripts\python -m pytest

# Frontend (92 tests)
cd frontend
npm test
npm run build
```

The frontend language test verifies that Türkçe, English, Deutsch, and Español expose the same UI keys and each provides a non-empty in-app guide.

---

## API URLs

| URL | Description |
|---|---|
| `http://127.0.0.1:5173` | Frontend dev server |
| `http://127.0.0.1:8000` | Backend API |
| `http://127.0.0.1:8000/health` | Backend health check |
| `ws://127.0.0.1:8000/ws/run` | WebSocket token stream |
| `http://127.0.0.1:8000/experiments` | `GET` list saved reports · `POST` save one |
| `http://127.0.0.1:8000/experiments/{id}` | `GET` full report · `DELETE` remove it |
| `http://127.0.0.1:8000/experiments/{id}/review` | `PATCH` manual report/row review verdicts and notes |
| `http://127.0.0.1:8000/experiments/{id}/csv` | Results table as CSV |

---

## Language Support

English · Türkçe · Deutsch · Español

Toggle language anytime from the language selector. UI copy and the in-app **How to Use** guide are stored together in separate modules:

```text
frontend/src/languages/
├── en.ts
├── tr.ts
├── de.ts
├── es.ts
└── types.ts
```

`frontend/src/i18n.ts` registers the languages and shared lookup helpers; `frontend/src/guide.ts` selects the guide for the active language.

---

## Security Note & Disclaimer

This tool is developed strictly for AI safety research and mechanistic interpretability. **Using jailbreaks on cloud API adapters (OpenAI / Anthropic / Gemini) may violate terms of service. Use at your own risk.**
