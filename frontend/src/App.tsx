import {
  Archive,
  BookOpen,
  BrainCircuit,
  Download,
  Gauge,
  ListChecks,
  MessageSquare,
  Pause,
  Plus,
  RotateCcw,
  Save,
  ShieldAlert,
  SlidersHorizontal,
  Swords,
  Trash2
} from "lucide-react";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ExperimentDiffView } from "./components/ExperimentDiff";
import { BenchmarkPanel } from "./components/tabs/BenchmarkPanel";
import { ChatTab } from "./components/tabs/ChatTab";
import { AnalysisTab } from "./components/tabs/AnalysisTab";
import { ComparePanel } from "./components/tabs/ComparePanel";
import { ExperimentsPanel } from "./components/tabs/ExperimentsPanel";
import { GuideTab } from "./components/tabs/GuideTab";
import { SettingsTab } from "./components/tabs/SettingsTab";
import { profileFor } from "./adapters";
import { buildInterventions, countRuleLayers, type LayerOp, type UIRule } from "./interventions";
import { diffExperiments, type ExperimentDiff } from "./diff";
import { commitStep, emptyFrame, emptyTrace, readLayerFrame, snapshot, type TimelineFrame } from "./timeline";
import {
  CSV_COLUMNS,
  benchmarkReport,
  compareReport,
  modelCompareReport,
  sweepReport,
  deleteExperiment,
  downloadCsv,
  downloadJson,
  listExperiments,
  loadExperiment,
  redactConfig,
  saveExperiment,
  updateExperimentReview,
  type DraftReport
} from "./experiments";
import { assessmentForVisibleText } from "./outputAssessment";
import { DEFAULT_STEERING, RESEARCH_PRESETS } from "./presets";
import { conceptLabel, safetyNote, safetyStateLabel, translations, type Language } from "./i18n";
import { API_BASE, WS_URL, runPromptWS } from "./app/runtime";
import {
  JAILBREAK_MODES,
  SAMPLE_JSONL,
  benchmarkVerdict,
  parseBenchmarkJsonl,
  parseNumberList
} from "./app/benchmarking";
import {
  MAX_HISTORY_MESSAGES,
  adapterDefaults,
  defaultActiveRule,
  defaultIntervention,
  fallbackModels,
  type MainTab
} from "./app/defaults";
import type {
  AdapterName,
  AttentionTrace,
  BenchmarkResult,
  BlackBoxMetrics,
  Candidate,
  ChatTurn,
  CompareResult,
  ConceptTrace,
  ExperimentReport,
  ExperimentSummary,
  HeadMap,
  JailbreakMode,
  LayerMetric,
  LensToken,
  ModelInfo,
  ManualVerdict,
  OutputPolicy,
  OutputAssessment,
  PromptCraftType,
  Quantization,
  RunRequest,
  SafetyTrace,
  SteeringOptions,
  StreamEvent,
  ThinkPhaseSummary,
  TimelineTrace,
  TokenLimitMode
} from "./types";

export default function App() {
  const [hasAcceptedTerms, setHasAcceptedTerms] = useState(false);
  const [language, setLanguage] = useState<Language>("en");
  const t = translations[language];
  const [prompt, setPrompt] = useState<string>(translations.en.defaultPrompt);
  const [adapter, setAdapter] = useState<AdapterName>("mock");
  const [model, setModel] = useState(adapterDefaults.mock);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>(fallbackModels);
  const outputPolicy: OutputPolicy = "raw";
  const [maxTokens, setMaxTokens] = useState(80);
  const [tokenLimitMode, setTokenLimitMode] = useState<TokenLimitMode>("fixed");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [assistantPrefill, setAssistantPrefill] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({
    openai: localStorage.getItem("apiKey_openai") || "",
    anthropic: localStorage.getItem("apiKey_anthropic") || "",
    gemini: localStorage.getItem("apiKey_gemini") || "",
  });
  const [jailbreak, setJailbreak] = useState(false);
  const [jailbreakMode, setJailbreakMode] = useState<JailbreakMode>("default");
  const [steering, setSteering] = useState<SteeringOptions>({ ...DEFAULT_STEERING });
  const [steeringTargetMode, setSteeringTargetMode] = useState<"automatic" | "window" | "layers" | "depths">("automatic");
  const [selectedPreset, setSelectedPreset] = useState("custom");
  const [useMlpAblation, setUseMlpAblation] = useState(true);
  const [useHelpfulnessBoost, setUseHelpfulnessBoost] = useState(true);
  const [useNormRegulation, setUseNormRegulation] = useState(true);
  const [useDiversionSuppression, setUseDiversionSuppression] = useState(true);
  const [promptCraft, setPromptCraft] = useState<PromptCraftType>("none");
  const [quantization, setQuantization] = useState<Quantization>("none");
  const [interventions, setInterventions] = useState<UIRule[]>([]);
  // Click-to-intervene on the Layer Activity grid: layer index → what to do to
  // it. The Settings rule stack still handles ranges like "10-25"; this is the
  // direct-manipulation path, next to where you actually see the effect.
  const [layerOps, setLayerOps] = useState<Record<number, LayerOp>>({});
  const [brushAction, setBrushAction] = useState<LayerOp["action"]>("mute");
  const [brushScale, setBrushScale] = useState(0.5);
  const [layerCount, setLayerCount] = useState<number>(28);

  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [generatedText, setGeneratedText] = useState("");
  const [emptyOutputNotice, setEmptyOutputNotice] = useState(false);
  const [layers, setLayers] = useState<LayerMetric[]>([]);
  const [topK, setTopK] = useState<Candidate[]>([]);
  const [entropy, setEntropy] = useState<number | null>(null);
  const [hallucinationRisk, setHallucinationRisk] = useState<number | null>(null);
  const [safety, setSafety] = useState<SafetyTrace | null>(null);
  const [concepts, setConcepts] = useState<ConceptTrace | null>(null);
  const [lens, setLens] = useState<LensToken[]>([]);
  const [headMap, setHeadMap] = useState<HeadMap | null>(null);
  const [mutedHeads, setMutedHeads] = useState<Set<string>>(new Set());
  const [attention, setAttention] = useState<AttentionTrace | null>(null);
  const [blackBoxMetrics, setBlackBoxMetrics] = useState<BlackBoxMetrics | null>(null);
  const [promptTokens, setPromptTokens] = useState<number | null>(null);
  const [outputTokens, setOutputTokens] = useState<number | null>(null);
  const [effectiveMaxTokens, setEffectiveMaxTokens] = useState<number | null>(null);
  const [contextLength, setContextLength] = useState<number | null>(null);
  const [hardwareSafeMaxTokens, setHardwareSafeMaxTokens] = useState<number | null>(null);
  const [outputAssessment, setOutputAssessment] = useState<OutputAssessment | null>(null);
  const [thinkPhase, setThinkPhase] = useState<ThinkPhaseSummary | null>(null);
  const [currentPhase, setCurrentPhase] = useState<"think" | "answer">("answer");
  const [refused, setRefused] = useState<boolean | null>(null);
  // Surfaced as a banner on every page. Before this, a failed run only wrote to
  // the runtime log, which lives on the Analysis page — an OOM looked like a
  // silent no-op from the chat.
  const [runError, setRunError] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<TimelineTrace | null>(null);
  // The timeline grows once per generated token. Accumulating in a ref and
  // flushing on a timer keeps a 1024-token run from forcing 1024 canvas redraws.
  const timelineRef = useRef<TimelineTrace>(emptyTrace());
  const frameRef = useRef<TimelineFrame>(emptyFrame());
  const timelineDirtyRef = useRef(false);
  const lastFlushRef = useRef(0);
  const [log, setLog] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatTurn[]>([]);
  const [mainTab, setMainTab] = useState<MainTab>("chat");
  // Experiment store
  const [savedExperiments, setSavedExperiments] = useState<ExperimentSummary[]>([]);
  const [experimentsError, setExperimentsError] = useState<string | null>(null);
  const [experimentsLoading, setExperimentsLoading] = useState(false);
  const [diff, setDiff] = useState<ExperimentDiff | null>(null);
  // The last run's request, kept so a report records exactly what was sent
  // rather than the sidebar's current (possibly edited) settings.
  const lastRequestRef = useRef<RunRequest | null>(null);
  // Benchmark state
  const [benchmarkJsonl, setBenchmarkJsonl] = useState(SAMPLE_JSONL);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([]);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkProgress, setBenchmarkProgress] = useState(0);
  const benchmarkAbortRef = useRef<AbortController | null>(null);
  // Compare state
  const [comparePrompt, setComparePrompt] = useState("");
  const [compareKind, setCompareKind] = useState<"modes" | "models" | "sweep">("modes");
  const [compareSecondModel, setCompareSecondModel] = useState("");
  const [sweepStrengths, setSweepStrengths] = useState("0.5, 0.75, 1.0, 1.25");
  const [compareResults, setCompareResults] = useState<CompareResult[]>([]);
  const [compareRunning, setCompareRunning] = useState(false);
  const compareAbortRef = useRef<AbortController | null>(null);
  const [pendingUserMessage, setPendingUserMessage] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const generatedTextRef = useRef<string>("");
  const chatScrollRef = useRef<HTMLDivElement | null>(null);
  // Mirror of `pendingUserMessage` for any closure that outlives the render in
  // which `startRun` was called — chiefly `socket.onmessage`, which is bound
  // once and therefore captures stale React state. Reads from this ref are
  // synchronous and always current.
  const pendingUserMessageRef = useRef<string | null>(null);

  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);

  useEffect(() => {
    const node = chatScrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages, pendingUserMessage, generatedText]);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => res.json())
      .then((data) => {
        if (data.boot_id && data.boot_id === localStorage.getItem("acceptedBootId")) {
          setHasAcceptedTerms(true);
        }
      })
      .catch(console.error);
  }, []);

  const handleAcceptTerms = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      if (data.boot_id) {
        localStorage.setItem("acceptedBootId", data.boot_id);
      }
    } catch (err) {
      console.error(err);
    }
    setHasAcceptedTerms(true);
  };

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/models`)
      .then((response) => response.json())
      .then((items: ModelInfo[]) => {
        if (!active || !Array.isArray(items) || items.length === 0) return;
        setAvailableModels(items);
        setModel((current) => {
          const stillExists = items.some((item) => item.adapter === adapter && item.id === current);
          return stillExists ? current : (items.find((item) => item.adapter === adapter)?.id ?? adapterDefaults[adapter]);
        });
      })
      .catch(() => {
        setAvailableModels(fallbackModels);
      });
    return () => {
      active = false;
    };
  }, [adapter]);

  useEffect(() => {
    if (apiKeys.openai !== undefined) localStorage.setItem("apiKey_openai", apiKeys.openai);
    if (apiKeys.anthropic !== undefined) localStorage.setItem("apiKey_anthropic", apiKeys.anthropic);
    if (apiKeys.gemini !== undefined) localStorage.setItem("apiKey_gemini", apiKeys.gemini);
  }, [apiKeys]);

  const runLabel = t.status[status];
  const modelOptions = useMemo(() => availableModels.filter((item) => item.adapter === adapter), [adapter, availableModels]);
  const selectedModel = modelOptions.find((item) => item.id === model);
  // Benchmark + Compare need per-token logits and a `refused` verdict, which
  // only the white-box engines emit; mock/ollama/legacy-hook produce no usable
  // signal, so those tabs gate on this.
  const whiteboxAdapter = adapter === "nnsight" || adapter === "pytorch";
  const adapterProfile = profileFor(adapter);
  const isApiAdapter = adapter === "openai" || adapter === "anthropic" || adapter === "gemini";
  // Single in-flight guard across all three tabs — the backend adapters are
  // singletons hooking the same model, so two concurrent runs would clobber
  // each other's hooks/telemetry.
  const busy = status === "running" || benchmarkRunning || compareRunning;
  const activeInterventionCount = useMemo(() => countRuleLayers(interventions), [interventions]);
  // Concept keys come from the backend bank; the display names are localised.
  const conceptLabels = useMemo(() => {
    const names = concepts?.names ?? [];
    return Object.fromEntries(names.map((name) => [name, conceptLabel(language, name)]));
  }, [concepts, language]);
  const dominantLayer = useMemo(() => {
    if (!layers.length) return null;
    return layers.reduce((best, item) => (item.activity > best.activity ? item : best), layers[0]);
  }, [layers]);

  function resetTrace() {
    setGeneratedText("");
    setEmptyOutputNotice(false);
    setLayers([]);
    setTopK([]);
    setEntropy(null);
    setHallucinationRisk(null);
    setSafety(null);
    setConcepts(null);
    setLens([]);
    setHeadMap(null);
    setAttention(null);
    setBlackBoxMetrics(null);
    setPromptTokens(null);
    setOutputTokens(null);
    setEffectiveMaxTokens(null);
    setContextLength(null);
    setHardwareSafeMaxTokens(null);
    setOutputAssessment(null);
    setThinkPhase(null);
    setCurrentPhase("answer");
    setRefused(null);
    setRunError(null);
    setTimeline(null);
    timelineRef.current = emptyTrace();
    frameRef.current = emptyFrame();
    timelineDirtyRef.current = false;
    setLog([]);
  }

  /** Publish the accumulated timeline. Throttled while streaming; `force` on the
   *  terminal events so the final token is never left out. */
  function flushTimeline(force = false) {
    if (!timelineDirtyRef.current) return;
    const now = performance.now();
    if (!force && now - lastFlushRef.current < 200) return;
    lastFlushRef.current = now;
    timelineDirtyRef.current = false;
    setTimeline(snapshot(timelineRef.current));
  }

  /** Click a layer: apply the current brush, or lift it if that layer already
   *  carries the same action. */
  function toggleLayerOp(layer: number) {
    setLayerOps((current) => {
      const next = { ...current };
      if (next[layer]?.action === brushAction) delete next[layer];
      else next[layer] = { action: brushAction, scale: brushAction === "mute" ? 1 : brushScale };
      return next;
    });
  }

  function toggleHead(layer: number, head: number) {
    const key = `${layer}:${head}`;
    setMutedHeads((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function appendLog(message: string) {
    setLog((items) => [message, ...items].slice(0, 80));
  }

  function handleAdapterChange(next: AdapterName) {
    setAdapter(next);
    setModel(availableModels.find((item) => item.adapter === next)?.id ?? adapterDefaults[next]);
    setCompareSecondModel("");
  }

  function updateIntervention(index: number, patch: Partial<UIRule>) {
    setInterventions((items) => items.map((item, itemIndex) => (itemIndex === index ? { ...item, ...patch } : item)));
  }

  function removeIntervention(index: number) {
    setInterventions((items) => items.filter((_, itemIndex) => itemIndex !== index));
  }

  function addInterventionRule() {
    setInterventions((items) => [...items, { ...defaultActiveRule }].slice(0, 128));
  }

  function updateSteering(patch: Partial<SteeringOptions>) {
    setSelectedPreset("custom");
    setSteering((current) => ({ ...current, ...patch }));
  }

  function applyPreset(id: string) {
    setSelectedPreset(id);
    const preset = RESEARCH_PRESETS.find((item) => item.id === id);
    if (!preset) return;
    setJailbreak(preset.jailbreak);
    setJailbreakMode(preset.mode);
    setSteering({ ...preset.steering });
    setSteeringTargetMode(preset.steering.target_layers.length ? "layers" : preset.steering.target_depths.length ? "depths" : preset.steering.use_depth_window ? "window" : "automatic");
  }

  function startRun() {
    const trimmed = prompt.trim();
    if (!trimmed || busy) return;
    resetTrace();
    setStatus("running");
    generatedTextRef.current = "";
    pendingUserMessageRef.current = trimmed;
    setPendingUserMessage(trimmed);
    setPrompt("");

    const flatInterventions = buildInterventions(interventions, layerOps, mutedHeads);

    // Ship the prior turns as history so the model can attend to its own past
    // replies. The current user turn travels in `prompt`; the backend appends
    // it. Cap at MAX_HISTORY_MESSAGES so an unbounded conversation never trips
    // the backend schema's max_length=64.
    const historyPayload: ChatTurn[] = messages.slice(-MAX_HISTORY_MESSAGES).map((item) => ({
      role: item.role,
      content: item.content
    }));

    const request: RunRequest = {
      prompt: trimmed,
      system_prompt: systemPrompt.trim() || null,
      assistant_prefill: assistantPrefill || null,
      adapter,
      model,
      response_language: language,
      output_policy: outputPolicy,
      max_new_tokens: maxTokens,
      token_limit_mode: tokenLimitMode,
      temperature,
      api_key: apiKeys[adapter] ?? "",
      prompt_craft: promptCraft,
      jailbreak,
      jailbreak_mode: jailbreakMode,
      use_mlp_ablation: useMlpAblation,
      use_helpfulness_boost: useHelpfulnessBoost,
      use_norm_regulation: useNormRegulation,
      use_diversion_suppression: useDiversionSuppression,
      steering,
      quantization,
      intervention: flatInterventions[0] ?? defaultIntervention,
      interventions: flatInterventions,
      history: historyPayload
    };
    lastRequestRef.current = request;

    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;
    socket.onopen = () => {
      socket.send(JSON.stringify(request));
      appendLog(`${t.logs.runStarted}: ${adapter} / ${model}`);
    };
    socket.onmessage = (message) => {
      try {
        const parsed = JSON.parse(message.data) as StreamEvent;
        handleEvent(parsed);
      } catch { /* malformed frame — skip */ }
    };
    socket.onerror = () => {
      setStatus("error");
      setRunError(t.backendUnreachable);
      appendLog(t.logs.websocketError);
    };
    socket.onclose = () => {
      socketRef.current = null;
      setStatus((current) => (current === "running" ? "done" : current));
    };
  }

  function newChat() {
    if (status === "running") {
      socketRef.current?.close();
      socketRef.current = null;
    }
    setMessages([]);
    pendingUserMessageRef.current = null;
    setPendingUserMessage(null);
    setPrompt("");
    resetTrace();
    setStatus("idle");
    generatedTextRef.current = "";
    appendLog(t.logs.newChat);
  }

  function commitPendingTurn() {
    // Commit whatever the in-flight turn produced — used by stop/error so the
    // pending user bubble doesn't get stuck and the next run's history stays
    // consistent with what the user saw.
    const pendingUser = pendingUserMessageRef.current;
    const partial = generatedTextRef.current.trim();
    if (pendingUser || partial) {
      setMessages((prev) => {
        const next = [...prev];
        if (pendingUser) next.push({ role: "user", content: pendingUser });
        if (partial) next.push({ role: "assistant", content: partial });
        return next;
      });
    }
    pendingUserMessageRef.current = null;
    setPendingUserMessage(null);
    generatedTextRef.current = "";
  }

  function stopRun() {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stop" }));
      window.setTimeout(() => socket.close(), 150);
    } else {
      socket?.close();
    }
    setStatus("done");
    flushTimeline(true);
    commitPendingTurn();
    appendLog(t.logs.runStopped);
  }

  // ── Experiments ───────────────────────────────────────────────────────────

  /** The live telemetry, frozen. Everything the panels render comes from here,
   *  so a restored report repopulates the dashboard exactly. */
  function currentRunReport(): DraftReport {
    const request = lastRequestRef.current;
    return {
      kind: "run",
      label: "",
      notes: "",
      config: request
        ? redactConfig(request)
        : { prompt: pendingUserMessageRef.current ?? "", adapter, model, jailbreak, jailbreak_mode: jailbreakMode },
      result: {
        generated_text: generatedText,
        refused,
        prompt_tokens: promptTokens,
        output_tokens: outputTokens,
        effective_max_tokens: effectiveMaxTokens,
        context_length: contextLength,
        hardware_safe_max_tokens: hardwareSafeMaxTokens,
        assessment: outputAssessment,
        status
      },
      telemetry: {
        layers,
        top_k: topK,
        entropy,
        hallucination_risk: hallucinationRisk,
        safety,
        lens,
        head_map: headMap,
        attention,
        think_phase: thinkPhase,
        layer_count: layerCount,
        messages,
        log,
        timeline,
        concepts
      },
      rows: []
    };
  }

  const refreshExperiments = useCallback(async () => {
    setExperimentsLoading(true);
    try {
      setSavedExperiments(await listExperiments());
      setExperimentsError(null);
    } catch (err) {
      setExperimentsError(String(err));
    } finally {
      setExperimentsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (mainTab === "experiments") void refreshExperiments();
  }, [mainTab, refreshExperiments]);

  // The rail hides Analysis for API adapters; without this, switching adapter
  // while on that page leaves the user staring at a tab they can't navigate
  // back to.
  useEffect(() => {
    if (mainTab === "analysis" && isApiAdapter) setMainTab("chat");
  }, [mainTab, isApiAdapter]);

  async function persist(draft: DraftReport) {
    try {
      const saved = await saveExperiment(draft);
      appendLog(`${t.experimentSaved}: ${saved.id}`);
      await refreshExperiments();
    } catch (err) {
      appendLog(`${t.experimentSaveFailed}: ${err}`);
      setExperimentsError(String(err));
    }
  }

  /** Repopulate every panel from a saved report and drop the user back on the
   *  chat tab. The run is not re-executed — this is a replay of the snapshot. */
  function restoreExperiment(report: ExperimentReport) {
    if (report.kind !== "run") {
      if (report.kind === "benchmark") {
        setBenchmarkResults(report.rows as unknown as BenchmarkResult[]);
        setMainTab("benchmark");
      } else {
        setCompareResults(report.rows as unknown as CompareResult[]);
        setMainTab("compare");
      }
      appendLog(`${t.experimentLoaded}: ${report.id}`);
      return;
    }

    const tel = report.telemetry ?? {};
    setLayers(tel.layers ?? []);
    setTopK(tel.top_k ?? []);
    setEntropy(tel.entropy ?? null);
    setHallucinationRisk(tel.hallucination_risk ?? null);
    setSafety(tel.safety ?? null);
    setLens(tel.lens ?? []);
    setHeadMap(tel.head_map ?? null);
    setAttention(tel.attention ?? null);
    setThinkPhase(tel.think_phase ?? null);
    if (typeof tel.layer_count === "number") setLayerCount(tel.layer_count);
    setMessages(tel.messages ?? []);
    setLog(tel.log ?? []);
    // Reports saved before the timeline existed simply have no `timeline` key.
    const savedTimeline = tel.timeline ?? null;
    setTimeline(savedTimeline);
    timelineRef.current = savedTimeline
      ? { steps: savedTimeline.steps.slice(), safetyMatrix: savedTimeline.safetyMatrix, layerCount: savedTimeline.layerCount }
      : emptyTrace();
    setConcepts(tel.concepts ?? null);
    setRunError(null);

    const result = report.result ?? {};
    setGeneratedText(String(result.generated_text ?? ""));
    generatedTextRef.current = "";
    setRefused(typeof result.refused === "boolean" ? result.refused : null);
    setPromptTokens(typeof result.prompt_tokens === "number" ? result.prompt_tokens : null);
    setOutputTokens(typeof result.output_tokens === "number" ? result.output_tokens : null);
    setEffectiveMaxTokens(typeof result.effective_max_tokens === "number" ? result.effective_max_tokens : null);
    setContextLength(typeof result.context_length === "number" ? result.context_length : null);
    setHardwareSafeMaxTokens(typeof result.hardware_safe_max_tokens === "number" ? result.hardware_safe_max_tokens : null);
    setOutputAssessment(result.assessment && typeof result.assessment === "object" ? result.assessment as unknown as OutputAssessment : null);
    setPendingUserMessage(null);
    pendingUserMessageRef.current = null;
    setStatus("done");
    setMainTab("chat");
    appendLog(`${t.experimentLoaded}: ${report.id}`);
  }

  async function openExperiment(id: string) {
    try {
      restoreExperiment(await loadExperiment(id));
    } catch (err) {
      setExperimentsError(String(err));
    }
  }

  async function removeExperiment(id: string) {
    try {
      await deleteExperiment(id);
      await refreshExperiments();
    } catch (err) {
      setExperimentsError(String(err));
    }
  }

  async function compareExperiments(idA: string, idB: string) {
    try {
      const [reportA, reportB] = await Promise.all([loadExperiment(idA), loadExperiment(idB)]);
      setDiff(diffExperiments(reportA, reportB));
      setExperimentsError(null);
    } catch (err) {
      setExperimentsError(String(err));
    }
  }

  async function downloadExperiment(id: string) {
    try {
      const report = await loadExperiment(id);
      downloadJson(report, `${report.id}.json`);
    } catch (err) {
      setExperimentsError(String(err));
    }
  }

  async function reviewExperiment(id: string, verdict: ManualVerdict, notes = "") {
    const summary = savedExperiments.find((item) => item.id === id);
    try {
      await updateExperimentReview(id, {
        verdict,
        category: summary?.output_category || "other",
        notes,
        reviewer: "local-manual-review"
      });
      await refreshExperiments();
    } catch (err) {
      setExperimentsError(String(err));
    }
  }

  /** Config shared by the benchmark and compare reports — the settings those
   *  batch runners actually pin (temperature 0, no interventions). */
  function batchConfig(extra: Record<string, unknown>): Record<string, unknown> {
    return {
      adapter,
      model,
      max_new_tokens: maxTokens,
      token_limit_mode: tokenLimitMode,
      temperature: 0,
      prompt_craft: promptCraft,
      quantization,
      use_mlp_ablation: useMlpAblation,
      use_helpfulness_boost: useHelpfulnessBoost,
      use_norm_regulation: useNormRegulation,
      use_diversion_suppression: useDiversionSuppression,
      steering,
      system_prompt: systemPrompt.trim() || null,
      assistant_prefill: assistantPrefill || null,
      ...extra
    };
  }

  async function unloadModels() {
    try {
      const r = await fetch("http://127.0.0.1:8000/unload", { method: "POST" });
      const data = await r.json();
      appendLog(`unload: ${(data.released || []).join(", ") || "(none)"}`);
    } catch (err) {
      appendLog(`unload failed: ${err}`);
    }
  }

  async function startBenchmark() {
    const cases = parseBenchmarkJsonl(benchmarkJsonl);
    if (!cases.length || busy || !whiteboxAdapter) return;
    setBenchmarkResults([]);
    setBenchmarkRunning(true);
    setBenchmarkProgress(0);
    const abort = new AbortController();
    benchmarkAbortRef.current = abort;
    const baseRequest: Omit<RunRequest, "prompt"> = { adapter, model, api_key: apiKeys[adapter] ?? "", system_prompt: systemPrompt.trim() || null, assistant_prefill: assistantPrefill || null, max_new_tokens: maxTokens, token_limit_mode: tokenLimitMode, temperature: 0, prompt_craft: promptCraft, jailbreak: false, jailbreak_mode: jailbreakMode, use_mlp_ablation: useMlpAblation, use_helpfulness_boost: useHelpfulnessBoost, use_norm_regulation: useNormRegulation, use_diversion_suppression: useDiversionSuppression, steering, quantization, output_policy: "raw", response_language: language, interventions: [], intervention: { enabled: false, target_type: "layer", layer: 0, head: null, action: "none", scale: 1 }, history: [] };
    for (let i = 0; i < cases.length; i++) {
      if (abort.signal.aborted) break;
      const c = cases[i];
      const run = await runPromptWS({ ...baseRequest, prompt: c.prompt } as RunRequest, abort.signal);
      const result: BenchmarkResult = { ...c, ...run, verdict: benchmarkVerdict(run.refused, c.expected_refusal, run.assessment) };
      setBenchmarkResults((prev) => [...prev, result]);
      setBenchmarkProgress(i + 1);
    }
    setBenchmarkRunning(false);
  }

  function stopBenchmark() {
    benchmarkAbortRef.current?.abort();
    setBenchmarkRunning(false);
  }

  async function startCompare() {
    const p = comparePrompt.trim();
    if (!p || busy || !whiteboxAdapter) return;
    setCompareResults([]);
    setCompareRunning(true);
    const abort = new AbortController();
    compareAbortRef.current = abort;
    const baseRequest: Omit<RunRequest, "jailbreak" | "jailbreak_mode"> = { prompt: p, adapter, model, api_key: apiKeys[adapter] ?? "", system_prompt: systemPrompt.trim() || null, assistant_prefill: assistantPrefill || null, max_new_tokens: maxTokens, token_limit_mode: tokenLimitMode, temperature: 0, prompt_craft: promptCraft, use_mlp_ablation: useMlpAblation, use_helpfulness_boost: useHelpfulnessBoost, use_norm_regulation: useNormRegulation, use_diversion_suppression: useDiversionSuppression, steering, quantization, output_policy: "raw", response_language: language, interventions: [], intervention: { enabled: false, target_type: "layer", layer: 0, head: null, action: "none", scale: 1 }, history: [] };
    if (compareKind === "models") {
      const targets = [model, compareSecondModel].filter((item, index, all) => item && all.indexOf(item) === index);
      for (const target of targets) {
        if (abort.signal.aborted) break;
        const run = await runPromptWS({ ...baseRequest, model: target, jailbreak: false, jailbreak_mode: "default" } as RunRequest, abort.signal);
        const label = availableModels.find((item) => item.adapter === adapter && item.id === target)?.label ?? target.split(/[\\/]/).pop() ?? target;
        setCompareResults((prev) => [...prev, { mode: label, jailbreak: false, ...run }]);
      }
      setCompareRunning(false);
      return;
    }
    if (compareKind === "sweep") {
      const values = parseNumberList(sweepStrengths).filter((value) => value >= 0 && value <= 5).slice(0, 20);
      for (const strength of values) {
        if (abort.signal.aborted) break;
        const run = await runPromptWS({ ...baseRequest, jailbreak: true, jailbreak_mode: jailbreakMode, steering: { ...steering, strength } } as RunRequest, abort.signal);
        setCompareResults((prev) => [...prev, { mode: `strength=${strength}`, jailbreak: true, ...run }]);
      }
      setCompareRunning(false);
      return;
    }
    // baseline first
    const baseRun = await runPromptWS({ ...baseRequest, jailbreak: false, jailbreak_mode: "default" } as RunRequest, abort.signal);
    setCompareResults([{ mode: "baseline", jailbreak: false, ...baseRun }]);
    // then every registered steering mode
    for (const mode of JAILBREAK_MODES) {
      if (abort.signal.aborted) break;
      const run = await runPromptWS({ ...baseRequest, jailbreak: true, jailbreak_mode: mode } as RunRequest, abort.signal);
      setCompareResults((prev) => [...prev, { mode, jailbreak: true, ...run }]);
    }
    setCompareRunning(false);
  }

  function stopCompare() {
    compareAbortRef.current?.abort();
    setCompareRunning(false);
  }

  function handleEvent(event: StreamEvent) {
    if (event.type === "run_started") {
      const data = event.data as { prompt_tokens?: number; layer_count?: number; effective_max_tokens?: number; context_length?: number | null; hardware_safe_max_tokens?: number | null };
      if (typeof data.prompt_tokens === "number") setPromptTokens(data.prompt_tokens);
      if (typeof data.layer_count === "number") setLayerCount(data.layer_count);
      if (typeof data.effective_max_tokens === "number") setEffectiveMaxTokens(data.effective_max_tokens);
      if (typeof data.context_length === "number") setContextLength(data.context_length);
      if (typeof data.hardware_safe_max_tokens === "number") setHardwareSafeMaxTokens(data.hardware_safe_max_tokens);
      appendLog(t.logs.modelRunnerOpened);
    }
    if (event.type === "layer_activity") {
      const data = event.data as { layers: LayerMetric[] };
      setLayers(data.layers);
      if (data.layers.length > 0) setLayerCount(data.layers.length);
      // Stash this step's per-layer safety; the `token` event commits it.
      Object.assign(frameRef.current, readLayerFrame(data.layers));
    }
    if (event.type === "prompt_crafted") {
      const data = event.data as { crafted_prompt: string };
      appendLog(`Prompt crafted: ${data.crafted_prompt.slice(0, 45)}...`);
    }
    if (event.type === "token") {
      const data = event.data as { generated_text: string; text: string; index?: number; phase?: "think" | "answer" };
      setGeneratedText(data.generated_text);
      generatedTextRef.current = data.generated_text;
      if (data.phase) setCurrentPhase(data.phase);

      // `token` is emitted last in each backend step, so every other frame for
      // this step has already landed in frameRef — commit them as one sample.
      commitStep(timelineRef.current, frameRef.current, data.text ?? "", data.index);
      timelineDirtyRef.current = true;
      flushTimeline();
    }

    if (event.type === "uncertainty") {
      const data = event.data as { entropy: number; hallucination_risk: number; top_k: Candidate[] };
      setEntropy(data.entropy);
      setHallucinationRisk(data.hallucination_risk);
      setTopK(data.top_k);
      frameRef.current.entropy = data.entropy;
      frameRef.current.halluc = data.hallucination_risk;
    }
    if (event.type === "safety_status") {
      const data = event.data as { message?: string };
      if (data.message) appendLog(data.message);
    }
    if (event.type === "safety_trace") {
      setSafety(event.data as unknown as SafetyTrace);
    }
    if (event.type === "concept_trace") {
      setConcepts(event.data as unknown as ConceptTrace);
    }
    if (event.type === "layer_lens") {
      setLens((event.data as { layers: LensToken[] }).layers);
    }
    if (event.type === "head_map") {
      setHeadMap(event.data as unknown as HeadMap);
    }
    if (event.type === "attention") {
      setAttention(event.data as unknown as AttentionTrace);
    }
    if (event.type === "black_box_metrics") {
      setBlackBoxMetrics(event.data as BlackBoxMetrics);
    }
    if (event.type === "think_phase") {
      const tp = event.data as unknown as ThinkPhaseSummary;
      setThinkPhase(tp);
      appendLog(`Think phase: ${tp.think_steps} think / ${tp.answer_steps} answer steps`);
    }
    if (event.type === "intervention") {
      const data = event.data as { count?: number };
      appendLog(`${t.logs.interventionArmed}: ${data.count ?? 1}`);
    }
    if (event.type === "error") {
      const data = event.data as { message: string };
      setStatus("error");
      setRunError(data.message);
      flushTimeline(true);
      appendLog(data.message);
      // Commit whatever the turn produced so the pending bubble doesn't get
      // stuck and the next run's history is correct.
      commitPendingTurn();
    }
    if (event.type === "run_completed") {
      setStatus("done");
      flushTimeline(true);
      const data = event.data as { refused?: boolean; jailbreak?: boolean; output_tokens?: number; generated_text?: string; assessment?: OutputAssessment; finish_reason?: string };
      if (typeof data.output_tokens === "number") setOutputTokens(data.output_tokens);
      // Prefer the backend's whole-sequence decode. Decoding one token at a
      // time can be empty for processor/control tokens even when the completed
      // sequence has visible text.
      const pendingUser = pendingUserMessageRef.current;
      const finalText = (data.generated_text || generatedTextRef.current || "").trim();
      const assessment = assessmentForVisibleText(data.assessment, finalText);
      setRefused(finalText ? data.refused ?? null : null);
      setOutputAssessment(assessment);
      if (assessment?.category) {
        appendLog(`${data.jailbreak ? "jailbreak" : "baseline"} → ${assessment.category}${data.finish_reason ? ` (${data.finish_reason})` : ""}`);
      }
      appendLog(t.logs.runCompleted);
      setGeneratedText(finalText);
      generatedTextRef.current = finalText;
      setEmptyOutputNotice(!finalText);
      if (pendingUser || finalText) {
        setMessages((prev) => {
          const next = [...prev];
          if (pendingUser) next.push({ role: "user", content: pendingUser });
          if (finalText) next.push({ role: "assistant", content: finalText });
          return next;
        });
      }
      pendingUserMessageRef.current = null;
      setPendingUserMessage(null);
    }
  }

  const navItems: Array<{ id: MainTab; icon: React.ReactNode; label: string; hidden?: boolean }> = [
    { id: "chat", icon: <MessageSquare size={18} />, label: t.navChat },
    { id: "analysis", icon: <BrainCircuit size={18} />, label: t.navAnalysis, hidden: isApiAdapter },
    { id: "benchmark", icon: <ListChecks size={18} />, label: t.tabBenchmark },
    { id: "compare", icon: <Swords size={18} />, label: t.tabCompare },
    { id: "experiments", icon: <Archive size={18} />, label: t.tabExperiments }
  ];

  const pageTitle: Record<MainTab, string> = {
    chat: t.navChat,
    analysis: t.navAnalysis,
    benchmark: t.tabBenchmark,
    compare: t.tabCompare,
    experiments: t.tabExperiments,
    settings: t.navSettings,
    guide: t.tabGuide
  };

  const modelLabel = selectedModel?.label ?? model.split(/[\\/]/).pop() ?? model;

  return (
    <>
      {!hasAcceptedTerms && (
        <div className="disclaimer-overlay">
          <div className="disclaimer-content">
            <h2 className="danger-text"><ShieldAlert size={26} /> {t.ui.ethicalTitle}</h2>
            <div className="disclaimer-body">
              <p>
                <strong>LLM Mind Visualizer &amp; Prompt Lab</strong> {t.ui.ethicalIntro}
              </p>
              <ul>
                <li>{t.ui.ethicalContent}</li>
                <li>{t.ui.ethicalAuthorized}</li>
                <li>{t.ui.ethicalApi}</li>
                <li>{t.ui.ethicalNoHarm}</li>
              </ul>
              <p className="disclaimer-footer">{t.ui.ethicalWarning}</p>
            </div>
            <button className="primary accept-btn" onClick={handleAcceptTerms}>
              {t.ui.ethicalAccept}
            </button>
          </div>
        </div>
      )}

      <div className={`shell${!hasAcceptedTerms ? " blurred-shell" : ""}`}>
        {/* ── Icon rail ─────────────────────────────────────────────────── */}
        <nav className="rail" aria-label={t.navChat}>
          <div className="rail-brand" title="LLM Mind Visualizer">
            <BrainCircuit size={20} />
          </div>
          <div className="rail-group">
            {navItems.filter((item) => !item.hidden).map((item) => (
              <button
                key={item.id}
                className={`rail-btn${mainTab === item.id ? " active" : ""}`}
                onClick={() => setMainTab(item.id)}
                title={item.label}
                aria-current={mainTab === item.id}
              >
                {item.icon}
                <span className="rail-tip">{item.label}</span>
              </button>
            ))}
          </div>
          <div className="rail-group rail-bottom">
            <button
              className={`rail-btn${mainTab === "guide" ? " active" : ""}`}
              onClick={() => setMainTab("guide")}
              title={t.tabGuide}
            >
              <BookOpen size={18} />
              <span className="rail-tip">{t.tabGuide}</span>
            </button>
            <button
              className={`rail-btn${mainTab === "settings" ? " active" : ""}`}
              onClick={() => setMainTab("settings")}
              title={t.navSettings}
            >
              <SlidersHorizontal size={18} />
              <span className="rail-tip">{t.navSettings}</span>
            </button>
          </div>
        </nav>

        {/* ── Page ──────────────────────────────────────────────────────── */}
        <div className="page">
          <header className="page-head">
            <div className="page-head-title">
              <h1>{pageTitle[mainTab]}</h1>
              {mainTab === "chat" ? <span className="page-head-sub">{modelLabel}</span> : null}
            </div>
            <div className="page-head-actions">
              {mainTab === "chat" && !isApiAdapter && safety ? (
                <button className="ghost stat-link" onClick={() => setMainTab("analysis")} title={t.viewAnalysis}>
                  <Gauge size={14} />
                  {t.safety} {Math.round(safety.score * 100)}%
                </button>
              ) : null}
              {mainTab === "chat" ? (
                <>
                  <button className="ghost" onClick={() => void persist(currentRunReport())} disabled={status === "running" || !generatedText} title={t.experimentSaveRunTitle}>
                    <Save size={14} /> {t.experimentSaveRun}
                  </button>
                  <button className="ghost" onClick={() => downloadJson(currentRunReport())} disabled={status === "running" || !generatedText} title={t.experimentDownloadTitle}>
                    <Download size={14} />
                  </button>
                  <button className="ghost" onClick={newChat} disabled={messages.length === 0 && !pendingUserMessage && !generatedText} title={t.newChatTitle}>
                    <Plus size={14} /> {t.newChat}
                  </button>
                </>
              ) : null}
              <span className={`status-dot ${status}`} title={runLabel}>
                <i />
                {runLabel}
              </span>
            </div>
          </header>

          {runError ? (
            <div className="error-banner" role="alert">
              <ShieldAlert size={15} />
              <div className="error-banner-body">
                <strong>{t.runFailed}</strong>
                <span>{runError}</span>
              </div>
              <button className="ghost icon" onClick={() => setRunError(null)} title={t.guideClose}>
                <Trash2 size={14} />
              </button>
            </div>
          ) : null}

          {mainTab === "analysis" && adapterProfile.reasonKey ? (
            <div className={`adapter-note tier-${adapterProfile.tier}`}>
              <ShieldAlert size={15} />
              <div>
              <strong>{t.adapterCapTitle}</strong>
              <span>{t[adapterProfile.reasonKey]}</span>
              </div>
            {adapter !== "pytorch" ? (
              <button className="ghost" onClick={() => handleAdapterChange("pytorch")}>
              {t.adapterCapSwitch}
              </button>
              ) : null}
            </div>
            ) : null}

          <div className="page-body">
            {mainTab === "chat" ? (
<ChatTab
                t={t}
                messages={messages}
                pendingUserMessage={pendingUserMessage}
                status={status}
                generatedText={generatedText}
                emptyOutputNotice={emptyOutputNotice}
                prompt={prompt}
                onPromptChange={setPrompt}
                onStart={startRun}
                onStop={stopRun}
                onOpenSettings={() => setMainTab("settings")}
                threadRef={chatScrollRef}
                modelLabel={modelLabel}
                jailbreak={jailbreak}
                jailbreakMode={jailbreakMode}
                onJailbreakChange={setJailbreak}
                isApiAdapter={isApiAdapter}
                activeRuleCount={activeInterventionCount + mutedHeads.size + Object.keys(layerOps).length}
                outputTokens={outputTokens}
                effectiveMaxTokens={effectiveMaxTokens}
                busy={busy}
                outputAssessment={outputAssessment}
              />
            ) : mainTab === "analysis" ? (
<AnalysisTab
                t={t}
                language={language}
                entropy={entropy}
                hallucinationRisk={hallucinationRisk}
                dominantLayer={dominantLayer}
                safety={safety}
                promptTokens={promptTokens}
                outputTokens={outputTokens}
                outputAssessment={outputAssessment}
                timeline={timeline}
                concepts={concepts}
                conceptLabels={conceptLabels}
                layerOps={layerOps}
                onClearLayerOps={() => setLayerOps({})}
                brushAction={brushAction}
                onBrushActionChange={setBrushAction}
                brushScale={brushScale}
                onBrushScaleChange={setBrushScale}
                layers={layers}
                layerCount={layerCount}
                onToggleLayer={toggleLayerOp}
                lens={lens}
                topK={topK}
                headMap={headMap}
                mutedHeads={mutedHeads}
                onToggleHead={toggleHead}
                attention={attention}
                thinkPhase={thinkPhase}
                currentPhase={currentPhase}
                blackBoxMetrics={blackBoxMetrics}
                log={log}
              />
            ) : mainTab === "benchmark" ? (
              <BenchmarkPanel
                t={t}
                jsonl={benchmarkJsonl}
                onJsonlChange={setBenchmarkJsonl}
                results={benchmarkResults}
                running={benchmarkRunning}
                progress={benchmarkProgress}
                total={parseBenchmarkJsonl(benchmarkJsonl).length}
                onRun={startBenchmark}
                onStop={stopBenchmark}
                onClear={() => setBenchmarkResults([])}
                onSave={() => void persist(benchmarkReport(benchmarkResults, batchConfig({ jailbreak: false })))}
                onExportJson={() => downloadJson(benchmarkReport(benchmarkResults, batchConfig({ jailbreak: false })))}
                onExportCsv={() => downloadCsv(benchmarkResults as unknown as Array<Record<string, unknown>>, CSV_COLUMNS.benchmark, "benchmark")}
                supported={whiteboxAdapter}
                busy={busy}
              />
            ) : mainTab === "compare" ? (
              <ComparePanel
                t={t}
                prompt={comparePrompt}
                onPromptChange={setComparePrompt}
                kind={compareKind}
                onKindChange={setCompareKind}
                secondModel={compareSecondModel}
                onSecondModelChange={setCompareSecondModel}
                modelOptions={modelOptions.filter((item) => item.id !== model)}
                sweepStrengths={sweepStrengths}
                onSweepStrengthsChange={setSweepStrengths}
                results={compareResults}
                running={compareRunning}
                onRun={startCompare}
                onStop={stopCompare}
                onSave={() => void persist(compareKind === "models" ? modelCompareReport(compareResults, batchConfig({ prompt: comparePrompt, models: [model, compareSecondModel] })) : compareKind === "sweep" ? sweepReport(compareResults, batchConfig({ prompt: comparePrompt, sweep_strengths: sweepStrengths })) : compareReport(compareResults, batchConfig({ prompt: comparePrompt })))}
                onExportJson={() => downloadJson(compareKind === "models" ? modelCompareReport(compareResults, batchConfig({ prompt: comparePrompt, models: [model, compareSecondModel] })) : compareKind === "sweep" ? sweepReport(compareResults, batchConfig({ prompt: comparePrompt, sweep_strengths: sweepStrengths })) : compareReport(compareResults, batchConfig({ prompt: comparePrompt })))}
                onExportCsv={() => downloadCsv(compareResults as unknown as Array<Record<string, unknown>>, CSV_COLUMNS.compare, "compare")}
                supported={whiteboxAdapter}
                busy={busy}
              />
            ) : mainTab === "experiments" ? (
              diff ? (
                <ExperimentDiffView diff={diff} t={t} onBack={() => setDiff(null)} />
              ) : (
                <ExperimentsPanel
                  t={t}
                  items={savedExperiments}
                  loading={experimentsLoading}
                  error={experimentsError}
                  onRefresh={() => void refreshExperiments()}
                  onOpen={(id) => void openExperiment(id)}
                  onDownload={(id) => void downloadExperiment(id)}
                  onDelete={(id) => void removeExperiment(id)}
                  onCompare={(idA, idB) => void compareExperiments(idA, idB)}
                  onReview={(id, verdict, notes) => void reviewExperiment(id, verdict, notes)}
                />
              )
            ) : mainTab === "settings" ? (
<SettingsTab
                t={t}
                adapter={adapter}
                onAdapterChange={handleAdapterChange}
                model={model}
                onModelChange={setModel}
                modelOptions={modelOptions}
                selectedModel={selectedModel}
                isApiAdapter={isApiAdapter}
                apiKey={apiKeys[adapter] ?? ""}
                onApiKeyChange={(value) => setApiKeys((current) => ({ ...current, [adapter]: value }))}
                whiteboxAdapter={whiteboxAdapter}
                onUnloadModels={() => void unloadModels()}
                running={status === "running"}
                tokenLimitMode={tokenLimitMode}
                onTokenLimitModeChange={setTokenLimitMode}
                maxTokens={maxTokens}
                onMaxTokensChange={setMaxTokens}
                temperature={temperature}
                onTemperatureChange={setTemperature}
                outputPolicy={outputPolicy}
                quantization={quantization}
                onQuantizationChange={setQuantization}
                effectiveMaxTokens={effectiveMaxTokens}
                contextLength={contextLength}
                hardwareSafeMaxTokens={hardwareSafeMaxTokens}
                systemPrompt={systemPrompt}
                onSystemPromptChange={setSystemPrompt}
                assistantPrefill={assistantPrefill}
                onAssistantPrefillChange={setAssistantPrefill}
                promptCraft={promptCraft}
                onPromptCraftChange={setPromptCraft}
                selectedPreset={selectedPreset}
                onApplyPreset={applyPreset}
                jailbreak={jailbreak}
                onJailbreakChange={setJailbreak}
                jailbreakMode={jailbreakMode}
                onJailbreakModeChange={setJailbreakMode}
                useMlpAblation={useMlpAblation}
                onUseMlpAblationChange={setUseMlpAblation}
                useHelpfulnessBoost={useHelpfulnessBoost}
                onUseHelpfulnessBoostChange={setUseHelpfulnessBoost}
                useNormRegulation={useNormRegulation}
                onUseNormRegulationChange={setUseNormRegulation}
                useDiversionSuppression={useDiversionSuppression}
                onUseDiversionSuppressionChange={setUseDiversionSuppression}
                steering={steering}
                steeringTargetMode={steeringTargetMode}
                onSteeringTargetModeChange={(value) => {
                  setSteeringTargetMode(value);
                  updateSteering({
                    target_layers: value === "layers" ? steering.target_layers : [],
                    target_depths: value === "depths" ? steering.target_depths : [],
                    use_depth_window: value === "window"
                  });
                }}
                onUpdateSteering={updateSteering}
                onResetSteering={() => {
                  setSelectedPreset("custom");
                  setSteeringTargetMode("automatic");
                  setSteering({ ...DEFAULT_STEERING });
                }}
                activeInterventionCount={activeInterventionCount}
                interventions={interventions}
                onAddIntervention={addInterventionRule}
                onUpdateIntervention={updateIntervention}
                onRemoveIntervention={removeIntervention}
                mutedHeadCount={mutedHeads.size}
                onClearMutedHeads={() => setMutedHeads(new Set())}
                language={language}
                onLanguageChange={setLanguage}
              />
            ) : (
              <GuideTab language={language} />
            )}
          </div>
        </div>
      </div>
    </>
  );
}

