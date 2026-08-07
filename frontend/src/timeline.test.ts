import { describe, expect, it } from "vitest";

import { commitStep, emptyFrame, emptyTrace, peakOf, readLayerFrame, snapshot } from "./timeline";
import type { LayerMetric } from "./types";

function layer(index: number, safety: number): LayerMetric {
  return { layer: index, activity: 0.5, safety, uncertainty: 0.1 };
}

describe("readLayerFrame", () => {
  it("extracts per-layer safety and its peak", () => {
    const frame = readLayerFrame([layer(0, 0.1), layer(1, 0.8), layer(2, 0.3)]);
    expect(frame.safety).toEqual([0.1, 0.8, 0.3]);
    expect(frame.peak).toBe(0.8);
  });

  it("handles an empty layer list", () => {
    expect(readLayerFrame([])).toEqual({ safety: [], peak: 0 });
  });

  it("never reports a negative peak", () => {
    expect(readLayerFrame([layer(0, -0.2)]).peak).toBe(0);
  });
});

describe("commitStep", () => {
  it("appends one sample per token", () => {
    const trace = emptyTrace();
    const frame = { ...emptyFrame(), safety: [0.1, 0.2], peak: 0.2, entropy: 1.5, halluc: 0.3 };

    commitStep(trace, frame, "The ", 0);
    commitStep(trace, frame, "sky", 1);

    expect(trace.steps).toHaveLength(2);
    expect(trace.steps[0]).toEqual({ step: 0, token: "The ", entropy: 1.5, halluc: 0.3, safety: 0.2 });
    expect(trace.steps[1].token).toBe("sky");
  });

  it("keeps the matrix aligned with the layer count", () => {
    const trace = emptyTrace();
    const frame = { ...emptyFrame(), safety: [0.1, 0.2, 0.3], peak: 0.3 };

    commitStep(trace, frame, "a", 0);
    commitStep(trace, frame, "b", 1);

    expect(trace.layerCount).toBe(3);
    expect(trace.safetyMatrix).toHaveLength(2);
    expect(trace.safetyMatrix.every((row) => row.length === 3)).toBe(true);
  });

  it("falls back to the running length when the backend sends no index", () => {
    const trace = emptyTrace();
    const frame = emptyFrame();
    commitStep(trace, frame, "a");
    commitStep(trace, frame, "b");
    expect(trace.steps.map((step) => step.step)).toEqual([0, 1]);
  });

  it("honours the backend's index when it is present", () => {
    const trace = emptyTrace();
    commitStep(trace, emptyFrame(), "a", 7);
    expect(trace.steps[0].step).toBe(7);
  });

  it("index 0 is not mistaken for a missing index", () => {
    // A plain `index || length` would turn step 0 into step 0 by luck, but the
    // same bug shows up as soon as the fallback is exercised — pin the type check.
    const trace = emptyTrace();
    trace.steps.push({ step: 99, token: "x", entropy: 0, halluc: 0, safety: 0 });
    commitStep(trace, emptyFrame(), "a", 0);
    expect(trace.steps[1].step).toBe(0);
  });

  it("still records a step when there is no layer frame", () => {
    // Black-box adapters emit tokens without per-layer telemetry; the step must
    // survive but must not push a ragged matrix row.
    const trace = emptyTrace();
    commitStep(trace, emptyFrame(), "a", 0);
    expect(trace.steps).toHaveLength(1);
    expect(trace.safetyMatrix).toHaveLength(0);
    expect(trace.layerCount).toBe(0);
  });

  it("records an empty token rather than dropping the step", () => {
    const trace = emptyTrace();
    commitStep(trace, emptyFrame(), "", 0);
    expect(trace.steps).toHaveLength(1);
    expect(trace.steps[0].token).toBe("");
  });
});

describe("snapshot", () => {
  it("returns a new wrapper so effects keyed on the trace re-run", () => {
    const trace = emptyTrace();
    commitStep(trace, { ...emptyFrame(), safety: [0.1], peak: 0.1 }, "a", 0);
    const first = snapshot(trace);
    expect(first).not.toBe(trace);
  });

  it("detaches the steps array so later commits don't mutate a published snapshot", () => {
    const trace = emptyTrace();
    commitStep(trace, emptyFrame(), "a", 0);
    const published = snapshot(trace);
    commitStep(trace, emptyFrame(), "b", 1);
    expect(published.steps).toHaveLength(1);
    expect(trace.steps).toHaveLength(2);
  });

  it("carries the layer count through", () => {
    const trace = emptyTrace();
    commitStep(trace, { ...emptyFrame(), safety: [0.1, 0.2], peak: 0.2 }, "a", 0);
    expect(snapshot(trace).layerCount).toBe(2);
  });
});

describe("peakOf", () => {
  it("returns the highest safety across the run", () => {
    const trace = emptyTrace();
    commitStep(trace, { ...emptyFrame(), peak: 0.2 }, "a", 0);
    commitStep(trace, { ...emptyFrame(), peak: 0.9 }, "b", 1);
    commitStep(trace, { ...emptyFrame(), peak: 0.4 }, "c", 2);
    expect(peakOf(trace)).toBe(0.9);
  });

  it("is null without a timeline", () => {
    expect(peakOf(null)).toBeNull();
    expect(peakOf(emptyTrace())).toBeNull();
  });
});
