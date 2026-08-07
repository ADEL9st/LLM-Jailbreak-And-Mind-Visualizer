from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app import experiments, run_control
from app.adapters.mock import MockAdapter
from app.adapters.ollama import OllamaAdapter
from app.adapters.openai_adapter import OpenaiAdapter
from app.adapters.anthropic_adapter import AnthropicAdapter
from app.adapters.gemini_adapter import GeminiAdapter
from app.model_compat import probe_config
from app.schemas import ModelInfo, RunRequest


def _try_load(import_path: str, class_name: str):
    try:
        mod = __import__(import_path, fromlist=[class_name])
        return getattr(mod, class_name)()
    except ImportError:
        return None
    except Exception:
        return None


app = FastAPI(title="LLM Mind Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

adapters = {
    "mock": MockAdapter(),
    "ollama": OllamaAdapter(),
    "openai": OpenaiAdapter(),
    "anthropic": AnthropicAdapter(),
    "gemini": GeminiAdapter(),
}
for key, import_path, class_name in (
    ("transformers", "app.adapters.transformers_hook", "TransformersHookAdapter"),
    ("pytorch", "app.adapters.pytorch_adapter", "PytorchAdapter"),
    ("nnsight", "app.adapters.nnsight_adapter", "NnsightAdapter"),
):
    instance = _try_load(import_path, class_name)
    if instance is not None:
        adapters[key] = instance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


import uuid

BOOT_ID = str(uuid.uuid4())

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "boot_id": BOOT_ID}


@app.get("/models")
async def models() -> list[ModelInfo]:
    detected = [
        ModelInfo(
            id="mock-qwen2.5-1.5b",
            label="Mock Qwen 1.5B Trace",
            adapter="mock",
            description="Deterministic simulated telemetry for UI and experiment flow.",
        ),
        ModelInfo(
            id="qwen2.5:1.5b",
            label="Ollama qwen2.5:1.5b",
            adapter="ollama",
            description="Black-box GGUF audit through Ollama.",
        ),
        # ── OpenAI ──────────────────────────────────────────────────────────
        ModelInfo(id="gpt-5.5",       label="GPT-5.5",       adapter="openai", description="OpenAI flagship — best reasoning & coding. 1M ctx, 128k output."),
        ModelInfo(id="gpt-5.4",       label="GPT-5.4",       adapter="openai", description="Powerful and cost-effective. 1M context window."),
        ModelInfo(id="gpt-5.4-mini",  label="GPT-5.4 Mini",  adapter="openai", description="Strong mini model for coding & agents. 400k context."),
        ModelInfo(id="gpt-5.4-nano",  label="GPT-5.4 Nano",  adapter="openai", description="Fastest and cheapest GPT-5.4 variant."),
        ModelInfo(id="gpt-4o",        label="GPT-4o",        adapter="openai", description="Previous flagship, multimodal, widely supported."),
        ModelInfo(id="gpt-4o-mini",   label="GPT-4o Mini",   adapter="openai", description="Cheap and fast; good for everyday tasks."),
        # ── Anthropic ───────────────────────────────────────────────────────
        ModelInfo(id="claude-fable-5",             label="Claude Fable 5",          adapter="anthropic", description="Most capable widely released Claude. 1M ctx, 128k output. $10/$50 MTok."),
        ModelInfo(id="claude-opus-4-8",            label="Claude Opus 4.8",         adapter="anthropic", description="Best Opus-tier: complex reasoning, agentic coding. 1M ctx. $5/$25 MTok."),
        ModelInfo(id="claude-sonnet-4-6",          label="Claude Sonnet 4.6",       adapter="anthropic", description="Best speed/intelligence balance. 1M ctx, 64k output. $3/$15 MTok."),
        ModelInfo(id="claude-haiku-4-5",           label="Claude Haiku 4.5",        adapter="anthropic", description="Fastest Claude with near-frontier intelligence. 200k ctx. $1/$5 MTok."),
        ModelInfo(id="claude-opus-4-7",            label="Claude Opus 4.7",         adapter="anthropic", description="Strong coding & vision. 1M ctx. $5/$25 MTok."),
        ModelInfo(id="claude-opus-4-6",            label="Claude Opus 4.6",         adapter="anthropic", description="Reliable & precise for enterprise workflows. 1M ctx."),
        ModelInfo(id="claude-sonnet-4-5",          label="Claude Sonnet 4.5",       adapter="anthropic", description="Fast and capable. 200k context."),
        ModelInfo(id="claude-opus-4-5",            label="Claude Opus 4.5",         adapter="anthropic", description="Previous Opus generation. 200k context."),
        # ── Google Gemini ────────────────────────────────────────────────────
        ModelInfo(id="gemini-3.5-flash",      label="Gemini 3.5 Flash",      adapter="gemini", description="Most intelligent Gemini for agentic & coding tasks (GA)."),
        ModelInfo(id="gemini-3.1-pro",        label="Gemini 3.1 Pro",        adapter="gemini", description="Advanced intelligence, complex problem-solving (Preview)."),
        ModelInfo(id="gemini-3.1-flash-lite", label="Gemini 3.1 Flash Lite", adapter="gemini", description="Cost-efficient, competitive performance (Stable)."),
        ModelInfo(id="gemini-2.5-pro",        label="Gemini 2.5 Pro",        adapter="gemini", description="Deep reasoning & coding. 1M token context (Stable)."),
        ModelInfo(id="gemini-2.5-flash",      label="Gemini 2.5 Flash",      adapter="gemini", description="Best price/performance for reasoning tasks (Stable)."),
        ModelInfo(id="gemini-2.5-flash-lite", label="Gemini 2.5 Flash Lite", adapter="gemini", description="Fastest and cheapest in the 2.5 family (Stable)."),
    ]
    detected.extend(_discover_transformers_models())
    return detected


def _discover_transformers_models() -> list[ModelInfo]:
    if not MODELS_DIR.exists():
        return []

    models_found: list[ModelInfo] = []
    for folder in sorted(item for item in MODELS_DIR.iterdir() if item.is_dir()):
        has_config = (folder / "config.json").exists()
        has_tokenizer = (folder / "tokenizer.json").exists() or (folder / "tokenizer_config.json").exists()
        has_weights = any(folder.glob("*.safetensors")) or any(folder.glob("*.bin"))
        if not (has_config and has_tokenizer and has_weights):
            continue

        model_id = f"../models/{folder.name}"
        label = _model_label(folder.name)
        try:
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            model_type = config.get("model_type", "")
        except (OSError, ValueError, TypeError):
            config = {}
            model_type = ""
        text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
        architectures = config.get("architectures") if isinstance(config.get("architectures"), list) else []
        profile = {
            "model_type": str(model_type),
            "architecture": str(architectures[0]) if architectures else "",
            "layer_count": _config_int(text_config, "num_hidden_layers"),
            "hidden_size": _config_int(text_config, "hidden_size"),
            "attention_heads": _config_int(text_config, "num_attention_heads"),
            "context_length": _config_int(text_config, "max_position_embeddings"),
            "dtype": str(text_config.get("torch_dtype") or config.get("torch_dtype") or ""),
            "size_bytes": sum(path.stat().st_size for path in folder.glob("*.safetensors")),
        }
        compatibility = probe_config(config)
        is_native_multimodal = compatibility.native_processor
        capabilities = ["white_box", *compatibility.capabilities]
        if "pytorch" in adapters:
            models_found.append(
                ModelInfo(
                    id=model_id,
                    label=f"{label} (pytorch)",
                    adapter="pytorch",
                    capabilities=capabilities,
                    compatibility=compatibility.as_dict(),
                    **profile,
                    description=(
                        "Native multimodal processor + PyTorch hooks — complete model inputs."
                        if is_native_multimodal
                        else "White-box via plain PyTorch hooks — faster, no hook leak, natural EOS."
                    ),
                )
            )
        # Native multimodal models need AutoProcessor and auxiliary model inputs.
        # The legacy string-only adapters silently discard those inputs and can
        # produce plausible-looking but invalid token streams, so do not offer
        # them for this architecture.
        if not is_native_multimodal and "nnsight" in adapters:
            models_found.append(
                ModelInfo(
                    id=model_id,
                    label=f"{label} (nnsight)",
                    adapter="nnsight",
                    capabilities=["white_box", "text", "nnsight"],
                    compatibility=compatibility.as_dict(),
                    **profile,
                    description="White-box via nnsight tracing — layer + head/neuron interventions.",
                )
            )
        if not is_native_multimodal and "transformers" in adapters:
            models_found.append(
                ModelInfo(
                    id=model_id,
                    label=f"{label} (hook v1)",
                    adapter="transformers",
                    capabilities=["white_box", "text", "legacy"],
                    compatibility=compatibility.as_dict(),
                    **profile,
                    description="Legacy white-box via manual forward hooks.",
                )
            )
    return models_found


def _model_label(folder_name: str) -> str:
    return folder_name.replace("-", " ").replace("_", " ").title()


def _config_int(config: dict, key: str) -> int | None:
    value = config.get(key)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


from app.prompt_craft import apply_prompt_craft

@app.post("/unload")
async def unload_models() -> dict:
    released = []
    for key, instance in adapters.items():
        if hasattr(instance, "unload"):
            try:
                instance.unload()
                released.append(key)
            except Exception:
                pass
    return {"released": released}


@app.get("/experiments")
async def list_experiments() -> list[experiments.ExperimentSummary]:
    return experiments.list_summaries()


@app.post("/experiments")
async def create_experiment(report: experiments.ExperimentReport) -> experiments.ExperimentReport:
    return experiments.save(report)


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> experiments.ExperimentReport:
    try:
        report = experiments.load(experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if report is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return report


@app.patch("/experiments/{experiment_id}/review")
async def review_experiment(
    experiment_id: str,
    update: experiments.ExperimentReviewUpdate,
) -> experiments.ExperimentReport:
    try:
        report = experiments.update_review(experiment_id, update)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if report is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return report


@app.get("/experiments/{experiment_id}/csv", response_class=PlainTextResponse)
async def get_experiment_csv(experiment_id: str) -> PlainTextResponse:
    try:
        report = experiments.load(experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if report is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    if not report.rows:
        raise HTTPException(status_code=400, detail="this experiment has no tabular rows")
    body = experiments.rows_to_csv(report.rows, experiments.columns_for(report.kind))
    return PlainTextResponse(
        body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{report.id}.csv"'},
    )


@app.delete("/experiments/{experiment_id}")
async def remove_experiment(experiment_id: str) -> dict:
    try:
        removed = experiments.delete(experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="experiment not found")
    return {"deleted": experiment_id}


BUSY_MESSAGE = (
    "Another run is already in progress on this server. Wait for it to finish or stop it, "
    "then try again."
)

# Adapters are module-level singletons that register hooks on one shared model
# instance, so two concurrent runs would clobber each other's hooks and
# interleave telemetry into both sockets. The UI's `busy` flag only guards a
# single tab; this guards a second tab, the benchmark CLI, or any other client.
_run_lock = asyncio.Lock()


@app.websocket("/ws/run")
async def run_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_text()
        request = RunRequest.model_validate_json(payload)

        # Reject rather than queue: a queued client would sit silent behind a
        # long generation with no way to tell that from a hang.
        if _run_lock.locked():
            await websocket.send_text(json.dumps({"type": "error", "ts": 0, "data": {"message": BUSY_MESSAGE}}))
            return

        # No await between the check above and the acquire below, so on the
        # single event loop this cannot race.
        async with _run_lock:
            await _stream_run(websocket, request)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_text(json.dumps({"type": "error", "ts": 0, "data": {"message": str(exc)}}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _stream_run(websocket: WebSocket, request: RunRequest) -> None:
    stream_iter = None
    cancel_event = threading.Event()

    async def watch_for_stop() -> None:
        try:
            while True:
                message = await websocket.receive_text()
                try:
                    payload = json.loads(message)
                except (TypeError, ValueError):
                    payload = {}
                if isinstance(payload, dict) and payload.get("type") == "stop":
                    cancel_event.set()
                    return
        except WebSocketDisconnect:
            cancel_event.set()

    stop_watcher = asyncio.create_task(watch_for_stop())
    try:
        if request.prompt_craft != "none":
            crafted = apply_prompt_craft(request.prompt, request.prompt_craft)
            request.prompt = crafted
            import time
            await websocket.send_text(json.dumps({
                "type": "prompt_crafted",
                "ts": time.perf_counter(),
                "data": {"crafted_prompt": crafted}
            }))

        adapter = adapters[request.adapter]
        with run_control.use(cancel_event):
            stream_iter = adapter.stream(request).__aiter__()
            while True:
                try:
                    item = await stream_iter.__anext__()
                except StopAsyncIteration:
                    break
                await websocket.send_text(json.dumps(item))
    finally:
        cancel_event.set()
        stop_watcher.cancel()
        try:
            await stop_watcher
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass
        # Always close the generator so the adapter's `finally` runs and its
        # forward hooks come off the model, even on disconnect mid-generation.
        if stream_iter is not None:
            try:
                await stream_iter.aclose()
            except Exception:
                pass
