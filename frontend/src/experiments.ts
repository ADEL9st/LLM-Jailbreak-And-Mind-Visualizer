/** Experiment persistence: build reports from live dashboard state, talk to the
 *  backend store, and hand files to the browser for download.
 *
 *  Kept out of App.tsx so the report shape lives in one place — a saved report
 *  is the tool's exchange format, and it should be readable without wading
 *  through the component. */

import type {
  BenchmarkResult,
  CompareResult,
  ExperimentReport,
  ExperimentSummary,
  ManualReview,
  RunRequest
} from "./types";

const API_BASE = "http://127.0.0.1:8000";

/** A report as the client builds it — the server assigns id/created_at/version. */
export type DraftReport = Pick<ExperimentReport, "kind" | "label" | "notes" | "config" | "result" | "telemetry" | "rows">;

/** Strip the API key before anything leaves the component. The backend redacts
 *  it too, but a local download never touches the backend. */
export function redactConfig(request: Partial<RunRequest>): Record<string, unknown> {
  const { api_key: _apiKey, ...rest } = request;
  return rest as Record<string, unknown>;
}

export async function saveExperiment(draft: DraftReport): Promise<ExperimentReport> {
  const response = await fetch(`${API_BASE}/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(draft)
  });
  if (!response.ok) throw new Error(`save failed (${response.status})`);
  return response.json();
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
  const response = await fetch(`${API_BASE}/experiments`);
  if (!response.ok) throw new Error(`list failed (${response.status})`);
  return response.json();
}

export async function loadExperiment(id: string): Promise<ExperimentReport> {
  const response = await fetch(`${API_BASE}/experiments/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error(`load failed (${response.status})`);
  return response.json();
}

export async function deleteExperiment(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/experiments/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`delete failed (${response.status})`);
}

export async function updateExperimentReview(
  id: string,
  review: ManualReview,
  rowReviews: Record<string, ManualReview> = {}
): Promise<ExperimentReport> {
  const response = await fetch(`${API_BASE}/experiments/${encodeURIComponent(id)}/review`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review, row_reviews: rowReviews })
  });
  if (!response.ok) throw new Error(`review update failed (${response.status})`);
  return response.json();
}

// ── Local download ──────────────────────────────────────────────────────────

function download(filename: string, body: string, mime: string) {
  const url = URL.createObjectURL(new Blob([body], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function stamp(): string {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

export function downloadJson(draft: DraftReport | ExperimentReport, filename?: string) {
  download(filename ?? `${draft.kind}-${stamp()}.json`, JSON.stringify(draft, null, 2), "application/json");
}

const BENCHMARK_COLUMNS = ["id", "category", "prompt", "expected_refusal", "refused", "verdict", "assessment", "finish_reason", "peak", "state", "elapsed", "text"];
const COMPARE_COLUMNS = ["mode", "jailbreak", "refused", "outcome", "assessment", "finish_reason", "peak", "state", "elapsed", "text"];

function csvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const raw = Array.isArray(value) || (typeof value === "object")
    ? JSON.stringify(value)
    // Newlines are flattened so one record stays one CSV line.
    : String(value).replace(/[\r\n]+/g, " ");
  return /[",]/.test(raw) ? `"${raw.replace(/"/g, '""')}"` : raw;
}

export function toCsv(rows: Array<Record<string, unknown>>, columns: string[]): string {
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((key) => csvCell(row[key])).join(","));
  return lines.join("\n") + "\n";
}

export function downloadCsv(rows: Array<Record<string, unknown>>, columns: string[], name: string) {
  download(`${name}-${stamp()}.csv`, toCsv(rows, columns), "text/csv;charset=utf-8");
}

// ── Report builders ─────────────────────────────────────────────────────────

export function benchmarkReport(results: BenchmarkResult[], config: Record<string, unknown>, label = ""): DraftReport {
  const count = (verdict: string) => results.filter((item) => item.verdict === verdict).length;
  const passed = count("PASS");
  return {
    kind: "benchmark",
    label,
    notes: "",
    config,
    result: {
      total: results.length,
      passed,
      bypass: count("FAIL:bypass"),
      overblock: count("FAIL:overblock"),
      review: count("REVIEW"),
      errors: count("ERROR"),
      pass_rate: results.length ? Number((passed / results.length).toFixed(4)) : 0
    },
    telemetry: {},
    rows: results as unknown as Array<Record<string, unknown>>
  };
}

export function compareReport(results: CompareResult[], config: Record<string, unknown>, label = ""): DraftReport {
  const baseline = results.find((item) => item.mode === "baseline");
  const flipped = results.filter((item) =>
    item.jailbreak
    && item.refused === false
    && item.assessment?.category === "complete_compliance"
    && item.assessment.coherent
    && !item.assessment.truncated
  );
  return {
    kind: "compare",
    label,
    notes: "",
    config,
    result: {
      modes_run: results.filter((item) => item.jailbreak).length,
      baseline_refused: baseline?.refused ?? null,
      baseline_peak: baseline?.peak ?? null,
      flipped_modes: flipped.map((item) => item.mode),
      flip_count: flipped.length
    },
    telemetry: {},
    rows: results as unknown as Array<Record<string, unknown>>
  };
}

export function modelCompareReport(results: CompareResult[], config: Record<string, unknown>, label = ""): DraftReport {
  return {
    kind: "model_compare",
    label,
    notes: "",
    config,
    result: {
      models_run: results.length,
      categories: Object.fromEntries(results.map((item) => [item.mode, item.assessment?.category ?? "unknown"])),
      manual_review_required: true
    },
    telemetry: {},
    rows: results as unknown as Array<Record<string, unknown>>
  };
}

export function sweepReport(results: CompareResult[], config: Record<string, unknown>, label = ""): DraftReport {
  return {
    kind: "sweep",
    label,
    notes: "",
    config,
    result: {
      runs: results.length,
      categories: Object.fromEntries(results.map((item) => [item.mode, item.assessment?.category ?? "unknown"])),
      manual_review_required: true
    },
    telemetry: {},
    rows: results as unknown as Array<Record<string, unknown>>
  };
}

export const CSV_COLUMNS = { benchmark: BENCHMARK_COLUMNS, compare: COMPARE_COLUMNS };
