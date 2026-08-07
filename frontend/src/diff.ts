/** Diffing two saved experiments.
 *
 *  The canonical use is baseline vs jailbreak on the same prompt: the per-layer
 *  safety delta shows *where* the intervention removed the refusal signal, which
 *  no single run can show on its own.
 *
 *  Pure functions over two reports — no fetching, no React. */

import type { ConceptScore, ExperimentReport, LayerMetric, TimelineTrace } from "./types";

export interface LayerDelta {
  layer: number;
  a: number;
  b: number;
  delta: number;
}

export interface ConceptDelta {
  name: string;
  a: number;
  b: number;
  delta: number;
  /** Layer each side peaked at, so a shift in *where* is visible too. */
  layerA: number | null;
  layerB: number | null;
}

export interface ScalarDelta {
  key: string;
  a: number | boolean | null;
  b: number | boolean | null;
  /** Only set when both sides are numeric. */
  delta: number | null;
}

export interface ExperimentDiff {
  a: ExperimentReport;
  b: ExperimentReport;
  /** False when the two runs have different layer counts — a cross-model diff,
   *  where per-layer deltas would be meaningless. */
  layersComparable: boolean;
  layerCount: number;
  samePrompt: boolean;
  safetyByLayer: LayerDelta[];
  activityByLayer: LayerDelta[];
  scalars: ScalarDelta[];
  timelineA: TimelineTrace | null;
  timelineB: TimelineTrace | null;
  /** Empty when either side predates the concept map or used an adapter that
   *  cannot produce one. */
  concepts: ConceptDelta[];
  textA: string;
  textB: string;
}

function layers(report: ExperimentReport): LayerMetric[] {
  return report.telemetry?.layers ?? [];
}

function pairByLayer(a: LayerMetric[], b: LayerMetric[], field: "safety" | "activity"): LayerDelta[] {
  const count = Math.min(a.length, b.length);
  const out: LayerDelta[] = [];
  for (let layer = 0; layer < count; layer += 1) {
    const left = a[layer]?.[field] ?? 0;
    const right = b[layer]?.[field] ?? 0;
    out.push({ layer, a: left, b: right, delta: right - left });
  }
  return out;
}

function num(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function bool(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function scalar(key: string, a: unknown, b: unknown): ScalarDelta {
  const left = typeof a === "boolean" ? a : num(a);
  const right = typeof b === "boolean" ? b : num(b);
  const delta = typeof left === "number" && typeof right === "number" ? right - left : null;
  return { key, a: left, b: right, delta };
}

/** Pair the two runs' ranked concept lists by name.
 *
 *  Concepts come from a fixed bank, but a report saved before a bank edit can
 *  carry a different set — so the union is taken and a side that lacks a
 *  concept scores 0 rather than being dropped. */
function pairConcepts(a: ExperimentReport, b: ExperimentReport): ConceptDelta[] {
  const left = a.telemetry?.concepts;
  const right = b.telemetry?.concepts;
  if (!left?.concepts?.length && !right?.concepts?.length) return [];

  const index = (report?: { concepts?: ConceptScore[] }) =>
    new Map((report?.concepts ?? []).map((item) => [item.name, item]));
  const mapA = index(left ?? undefined);
  const mapB = index(right ?? undefined);

  const names = [...new Set([...mapA.keys(), ...mapB.keys()])];
  return names
    .map((name) => {
      const scoreA = mapA.get(name)?.score ?? 0;
      const scoreB = mapB.get(name)?.score ?? 0;
      return {
        name,
        a: scoreA,
        b: scoreB,
        delta: scoreB - scoreA,
        layerA: mapA.get(name)?.layer ?? null,
        layerB: mapB.get(name)?.layer ?? null
      };
    })
    .sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta));
}

function safetyScore(report: ExperimentReport): number | null {
  const safety = report.telemetry?.safety;
  return safety && typeof safety.score === "number" ? safety.score : null;
}

/** Peak safety across the whole generation, which is a stronger signal than the
 *  final frame's score — a run can spike at token 0 and settle low. */
export function peakSafety(report: ExperimentReport): number | null {
  const steps = report.telemetry?.timeline?.steps;
  if (steps && steps.length) return Math.max(...steps.map((step) => step.safety));
  return safetyScore(report);
}

export function diffExperiments(a: ExperimentReport, b: ExperimentReport): ExperimentDiff {
  const layersA = layers(a);
  const layersB = layers(b);
  const layersComparable = layersA.length > 0 && layersA.length === layersB.length;

  return {
    a,
    b,
    layersComparable,
    layerCount: Math.min(layersA.length, layersB.length),
    samePrompt: String(a.config?.prompt ?? "") === String(b.config?.prompt ?? ""),
    safetyByLayer: pairByLayer(layersA, layersB, "safety"),
    activityByLayer: pairByLayer(layersA, layersB, "activity"),
    scalars: [
      scalar("refused", a.result?.refused, b.result?.refused),
      scalar("safety", safetyScore(a), safetyScore(b)),
      scalar("peakSafety", peakSafety(a), peakSafety(b)),
      scalar("outputTokens", a.result?.output_tokens, b.result?.output_tokens),
      scalar("promptTokens", a.result?.prompt_tokens, b.result?.prompt_tokens),
    ],
    timelineA: a.telemetry?.timeline ?? null,
    timelineB: b.telemetry?.timeline ?? null,
    concepts: pairConcepts(a, b),
    textA: String(a.result?.generated_text ?? ""),
    textB: String(b.result?.generated_text ?? ""),
  };
}

/** Layers where the two runs diverge most, strongest first — the answer to
 *  "where did the intervention actually land". */
export function biggestMovers(deltas: LayerDelta[], limit = 5): LayerDelta[] {
  return [...deltas].sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta)).slice(0, limit);
}

/** One-line description of a report, for labelling the two sides. */
export function describeRun(report: ExperimentReport): string {
  const config = report.config ?? {};
  if (config.jailbreak) return `jailbreak: ${String(config.jailbreak_mode || "default")}`;
  return "baseline";
}

export { bool };
