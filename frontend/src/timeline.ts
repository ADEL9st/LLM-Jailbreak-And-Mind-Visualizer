/** Accumulating the per-token timeline.
 *
 *  The backend emits several events per generation step and finishes each step
 *  with `token`. These helpers hold the partial frame between those events and
 *  commit it as one sample, so App.tsx's event handler stays a dispatcher and
 *  this logic is testable on its own.
 */

import type { LayerMetric, TimelineTrace } from "./types";

/** The in-progress step: filled by the `layer_activity` and `uncertainty`
 *  events, consumed by `token`. */
export interface TimelineFrame {
  safety: number[];
  peak: number;
  entropy: number;
  halluc: number;
}

export function emptyTrace(): TimelineTrace {
  return { steps: [], safetyMatrix: [], layerCount: 0 };
}

export function emptyFrame(): TimelineFrame {
  return { safety: [], peak: 0, entropy: 0, halluc: 0 };
}

/** Per-layer safety plus its peak, from a `layer_activity` payload. */
export function readLayerFrame(layers: LayerMetric[]): { safety: number[]; peak: number } {
  const safety = layers.map((item) => item.safety);
  return { safety, peak: safety.reduce((best, value) => Math.max(best, value), 0) };
}

/**
 * Append one generation step. Mutates `trace` — it is held in a ref and can
 * grow to 1024 steps, so copying on every token would be wasteful.
 *
 * `index` comes from the backend; when it is missing we fall back to the
 * current length so the step numbering stays dense either way.
 */
export function commitStep(
  trace: TimelineTrace,
  frame: TimelineFrame,
  token: string,
  index?: number
): TimelineTrace {
  trace.steps.push({
    step: typeof index === "number" ? index : trace.steps.length,
    token,
    entropy: frame.entropy,
    halluc: frame.halluc,
    safety: frame.peak
  });
  // A black-box or mid-run adapter may send a token with no layer frame; keep
  // the matrix rectangular by only appending when there is a row to append.
  if (frame.safety.length) {
    trace.safetyMatrix.push(frame.safety);
    trace.layerCount = frame.safety.length;
  }
  return trace;
}

/** A snapshot for React state. The wrapper is new every time so effects keyed
 *  on the trace (the heatmap canvas) re-run; `safetyMatrix` is shared because
 *  its rows are never mutated after being pushed. */
export function snapshot(trace: TimelineTrace): TimelineTrace {
  return { steps: trace.steps.slice(), safetyMatrix: trace.safetyMatrix, layerCount: trace.layerCount };
}

/** Peak safety across a whole run, or null when there is no timeline. */
export function peakOf(trace: TimelineTrace | null): number | null {
  if (!trace || !trace.steps.length) return null;
  return trace.steps.reduce((best, step) => Math.max(best, step.safety), 0);
}
