import { describe, expect, it } from "vitest";

import { CSV_COLUMNS, benchmarkReport, compareReport, modelCompareReport, redactConfig, sweepReport, toCsv } from "./experiments";
import type { BenchmarkResult, CompareResult, RunRequest } from "./types";

describe("redactConfig", () => {
  it("drops the API key before a report can leave the browser", () => {
    // The backend strips it too, but a local JSON download never goes through
    // the backend — this is the only guard on that path.
    const config = redactConfig({ prompt: "p", api_key: "sk-SECRET", adapter: "openai" } as Partial<RunRequest>);
    expect(config).not.toHaveProperty("api_key");
    expect(JSON.stringify(config)).not.toContain("sk-SECRET");
  });

  it("keeps everything else", () => {
    const config = redactConfig({ prompt: "p", jailbreak: true, max_new_tokens: 40 } as Partial<RunRequest>);
    expect(config).toMatchObject({ prompt: "p", jailbreak: true, max_new_tokens: 40 });
  });

  it("is fine with no key present", () => {
    expect(redactConfig({ prompt: "p" } as Partial<RunRequest>)).toEqual({ prompt: "p" });
  });
});

describe("toCsv", () => {
  it("writes a header row followed by one line per record", () => {
    const body = toCsv([{ a: 1, b: 2 }, { a: 3, b: 4 }], ["a", "b"]);
    expect(body.trim().split("\n")).toEqual(["a,b", "1,2", "3,4"]);
  });

  it("quotes commas and doubles embedded quotes", () => {
    const body = toCsv([{ text: 'a, b "q"' }], ["text"]);
    expect(body.split("\n")[1]).toBe('"a, b ""q"""');
  });

  it("flattens newlines so one record stays one line", () => {
    const body = toCsv([{ text: "line1\nline2\r\nline3" }], ["text"]);
    expect(body.trim().split("\n")).toHaveLength(2);
    expect(body).toContain("line1 line2 line3");
  });

  it("JSON-encodes arrays and objects", () => {
    const body = toCsv([{ errors: ["boom", "bang"] }], ["errors"]);
    expect(body.split("\n")[1]).toBe('"[""boom"",""bang""]"');
  });

  it("renders null and undefined as empty cells, not the strings", () => {
    const body = toCsv([{ a: null, b: undefined }], ["a", "b"]);
    expect(body.split("\n")[1]).toBe(",");
  });

  it("emits an empty cell for a column a row does not have", () => {
    expect(toCsv([{ a: 1 }], ["a", "missing"]).split("\n")[1]).toBe("1,");
  });

  it("keeps false and 0 rather than blanking them", () => {
    expect(toCsv([{ a: false, b: 0 }], ["a", "b"]).split("\n")[1]).toBe("false,0");
  });
});

const benchRow = (overrides: Partial<BenchmarkResult> = {}): BenchmarkResult => ({
  id: "b-001",
  category: "harm",
  prompt: "p",
  expected_refusal: true,
  peak: 0.8,
  state: "refusal_locked",
  refused: true,
  text: "no",
  errors: [],
  elapsed: 1,
  verdict: "PASS",
  ...overrides,
});

describe("benchmarkReport", () => {
  it("tallies verdicts and the pass rate", () => {
    const report = benchmarkReport(
      [
        benchRow(),
        benchRow({ id: "b-002", verdict: "FAIL:bypass" }),
        benchRow({ id: "b-003", verdict: "FAIL:overblock" }),
        benchRow({ id: "b-004", verdict: "ERROR" }),
      ],
      { adapter: "pytorch" }
    );
    expect(report.kind).toBe("benchmark");
    expect(report.result).toEqual({ total: 4, passed: 1, bypass: 1, overblock: 1, review: 0, errors: 1, pass_rate: 0.25 });
    expect(report.rows).toHaveLength(4);
  });

  it("does not divide by zero on an empty run", () => {
    expect(benchmarkReport([], {}).result.pass_rate).toBe(0);
  });

  it("carries the config through", () => {
    expect(benchmarkReport([], { adapter: "nnsight", model: "m" }).config).toMatchObject({ adapter: "nnsight" });
  });
});

const compareRow = (overrides: Partial<CompareResult> = {}): CompareResult => ({
  mode: "default",
  jailbreak: true,
  peak: 0.2,
  state: "clear",
  refused: false,
  text: "sure",
  elapsed: 1,
  errors: [],
  assessment: {
    category: "complete_compliance",
    content_status: "compliance",
    refusal_style: "none",
    truncated: false,
    complete: true,
    coherent: true,
    manual_review_required: true,
    legacy_outcome: "compliance"
  },
  ...overrides,
});

describe("compareReport", () => {
  it("records the baseline and which modes flipped the refusal", () => {
    const report = compareReport(
      [
        compareRow({ mode: "baseline", jailbreak: false, refused: true, peak: 0.9 }),
        compareRow({ mode: "default", refused: false }),
        compareRow({ mode: "surgical", refused: false }),
        compareRow({ mode: "broker_math", refused: true }),
      ],
      { prompt: "p" }
    );
    expect(report.result.baseline_refused).toBe(true);
    expect(report.result.baseline_peak).toBe(0.9);
    expect(report.result.flipped_modes).toEqual(["default", "surgical"]);
    expect(report.result.flip_count).toBe(2);
    expect(report.result.modes_run).toBe(3);
  });

  it("does not count a mode that errored (refused === null) as a flip", () => {
    const report = compareReport([compareRow({ refused: null })], {});
    expect(report.result.flip_count).toBe(0);
  });

  it("copes with no baseline row", () => {
    const report = compareReport([compareRow()], {});
    expect(report.result.baseline_refused).toBeNull();
  });
});

describe("CSV_COLUMNS", () => {
  it("exposes a column order for each tabular report kind", () => {
    expect(CSV_COLUMNS.benchmark[0]).toBe("id");
    expect(CSV_COLUMNS.compare[0]).toBe("mode");
  });
});

describe("research matrix reports", () => {
  it("keeps model comparisons separate from steering-mode flips", () => {
    const report = modelCompareReport([compareRow({ mode: "Qwen3" }), compareRow({ mode: "Qwen3.5" })], { prompt: "p" });
    expect(report.kind).toBe("model_compare");
    expect(report.result).toMatchObject({ models_run: 2, manual_review_required: true });
  });

  it("stores a parameter sweep as its own provenance-bearing kind", () => {
    const report = sweepReport([compareRow({ mode: "strength=0.5" })], { sweep_strengths: "0.5" });
    expect(report.kind).toBe("sweep");
    expect(report.rows).toHaveLength(1);
  });
});
