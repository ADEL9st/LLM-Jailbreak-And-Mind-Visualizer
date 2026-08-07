import { describe, expect, it } from "vitest";

import { biggestMovers, describeRun, diffExperiments, peakSafety } from "./diff";
import type { ExperimentReport, LayerMetric } from "./types";

function layers(safety: number[], activity: number[] = []): LayerMetric[] {
  return safety.map((value, index) => ({
    layer: index,
    activity: activity[index] ?? 0,
    safety: value,
    uncertainty: 0,
  }));
}

function report(overrides: Partial<ExperimentReport> = {}): ExperimentReport {
  return {
    id: "20260101-000000-aaaaaa",
    created_at: "2026-01-01T00:00:00+00:00",
    version: 1,
    kind: "run",
    label: "",
    notes: "",
    config: { prompt: "same prompt", model: "../models/qwen", jailbreak: false },
    result: { refused: true, generated_text: "I cannot help.", output_tokens: 10, prompt_tokens: 5 },
    telemetry: { layers: layers([0.2, 0.8]), safety: { score: 0.8, state: "refusal_locked", first_trigger_layer: 1, locked_layer: 1, notes: "" } },
    rows: [],
    ...overrides,
  };
}

describe("diffExperiments", () => {
  it("computes B − A per layer", () => {
    const a = report({ telemetry: { layers: layers([0.2, 0.9, 0.5]) } });
    const b = report({ telemetry: { layers: layers([0.1, 0.3, 0.5]) } });

    const diff = diffExperiments(a, b);
    expect(diff.safetyByLayer.map((item) => Number(item.delta.toFixed(2)))).toEqual([-0.1, -0.6, 0]);
    expect(diff.safetyByLayer[1]).toMatchObject({ layer: 1, a: 0.9, b: 0.3 });
  });

  it("diffs activity as well as safety", () => {
    const a = report({ telemetry: { layers: layers([0, 0], [0.1, 0.2]) } });
    const b = report({ telemetry: { layers: layers([0, 0], [0.4, 0.2]) } });
    expect(diffExperiments(a, b).activityByLayer.map((item) => Number(item.delta.toFixed(2)))).toEqual([0.3, 0]);
  });

  it("flags a cross-model pair as incomparable", () => {
    const a = report({ telemetry: { layers: layers([0.1, 0.2]) } });
    const b = report({ telemetry: { layers: layers([0.1, 0.2, 0.3, 0.4]) } });
    const diff = diffExperiments(a, b);
    expect(diff.layersComparable).toBe(false);
    // Still pairs what it can, so the caller decides whether to render it.
    expect(diff.safetyByLayer).toHaveLength(2);
  });

  it("treats a run with no layer telemetry as incomparable rather than crashing", () => {
    const diff = diffExperiments(report({ telemetry: {} }), report());
    expect(diff.layersComparable).toBe(false);
    expect(diff.safetyByLayer).toEqual([]);
  });

  it("detects differing prompts", () => {
    const a = report({ config: { prompt: "one" } });
    const b = report({ config: { prompt: "two" } });
    expect(diffExperiments(a, b).samePrompt).toBe(false);
    expect(diffExperiments(a, report({ config: { prompt: "one" } })).samePrompt).toBe(true);
  });

  it("carries both outputs through", () => {
    const a = report({ result: { generated_text: "refused" } });
    const b = report({ result: { generated_text: "complied" } });
    const diff = diffExperiments(a, b);
    expect(diff.textA).toBe("refused");
    expect(diff.textB).toBe("complied");
  });

  it("falls back to an empty string when a report has no output", () => {
    expect(diffExperiments(report({ result: {} }), report({ result: {} })).textA).toBe("");
  });
});

describe("scalar deltas", () => {
  const find = (diff: ReturnType<typeof diffExperiments>, key: string) =>
    diff.scalars.find((item) => item.key === key)!;

  it("subtracts numeric metrics", () => {
    const a = report({ result: { output_tokens: 10 }, telemetry: { safety: { score: 0.8 } } as never });
    const b = report({ result: { output_tokens: 40 }, telemetry: { safety: { score: 0.15 } } as never });
    const diff = diffExperiments(a, b);
    expect(find(diff, "outputTokens").delta).toBe(30);
    expect(find(diff, "safety").delta).toBeCloseTo(-0.65);
  });

  it("keeps booleans as booleans with no delta", () => {
    const a = report({ result: { refused: true } });
    const b = report({ result: { refused: false } });
    const refused = find(diffExperiments(a, b), "refused");
    expect(refused.a).toBe(true);
    expect(refused.b).toBe(false);
    expect(refused.delta).toBeNull();
  });

  it("reports null rather than 0 when a value is missing", () => {
    const diff = diffExperiments(report({ result: {} }), report({ result: { output_tokens: 5 } }));
    const tokens = find(diff, "outputTokens");
    expect(tokens.a).toBeNull();
    expect(tokens.delta).toBeNull();
  });
});

describe("peakSafety", () => {
  it("prefers the timeline peak over the final frame", () => {
    // The refusal decision spikes at token 0 and decays, so the last frame's
    // score understates the run — this is the whole reason the field exists.
    const withTimeline = report({
      telemetry: {
        safety: { score: 0.15 } as never,
        timeline: {
          steps: [
            { step: 0, token: "I", entropy: 1, halluc: 0, safety: 1.0 },
            { step: 1, token: " can", entropy: 1, halluc: 0, safety: 0.3 },
          ],
          safetyMatrix: [],
          layerCount: 0,
        },
      },
    });
    expect(peakSafety(withTimeline)).toBe(1.0);
  });

  it("falls back to the final safety score without a timeline", () => {
    expect(peakSafety(report())).toBe(0.8);
  });

  it("is null when there is neither", () => {
    expect(peakSafety(report({ telemetry: {} }))).toBeNull();
  });
});

describe("biggestMovers", () => {
  it("ranks by absolute change, so a large drop outranks a small rise", () => {
    const deltas = [
      { layer: 0, a: 0, b: 0, delta: 0.1 },
      { layer: 1, a: 0, b: 0, delta: -0.9 },
      { layer: 2, a: 0, b: 0, delta: 0.4 },
    ];
    expect(biggestMovers(deltas, 2).map((item) => item.layer)).toEqual([1, 2]);
  });

  it("does not mutate its input", () => {
    const deltas = [
      { layer: 0, a: 0, b: 0, delta: 0.1 },
      { layer: 1, a: 0, b: 0, delta: -0.9 },
    ];
    biggestMovers(deltas);
    expect(deltas[0].layer).toBe(0);
  });
});

describe("describeRun", () => {
  it("names the jailbreak mode", () => {
    expect(describeRun(report({ config: { jailbreak: true, jailbreak_mode: "surgical" } }))).toBe("jailbreak: surgical");
  });

  it("defaults the mode name when it is missing", () => {
    expect(describeRun(report({ config: { jailbreak: true } }))).toBe("jailbreak: default");
  });

  it("calls a non-jailbreak run baseline", () => {
    expect(describeRun(report())).toBe("baseline");
  });
});
