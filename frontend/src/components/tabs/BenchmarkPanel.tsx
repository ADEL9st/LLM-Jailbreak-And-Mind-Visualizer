import React, { useState } from "react";
import { Download, ListChecks, Play, RotateCcw, Save, Square } from "lucide-react";
import { SAMPLE_JSONL } from "../../app/benchmarking";
import { KNOWLEDGE_JSONL } from "../../presets";
import type { Translation } from "../../i18n";
import type { BenchmarkResult } from "../../types";

interface BenchmarkPanelProps {
  t: Translation;
  jsonl: string;
  onJsonlChange: (value: string) => void;
  results: BenchmarkResult[];
  running: boolean;
  progress: number;
  total: number;
  onRun: () => void;
  onStop: () => void;
  onClear: () => void;
  onSave: () => void;
  onExportJson: () => void;
  onExportCsv: () => void;
  supported: boolean;
  busy: boolean;
}

export function BenchmarkPanel({
  t, jsonl, onJsonlChange, results, running, progress, total, onRun, onStop,
  onClear, onSave, onExportJson, onExportCsv, supported, busy
}: BenchmarkPanelProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const pass = results.filter((result) => result.verdict === "PASS").length;
  const bypass = results.filter((result) => result.verdict === "FAIL:bypass").length;
  const overblock = results.filter((result) => result.verdict === "FAIL:overblock").length;
  const errors = results.filter((result) => result.verdict === "ERROR").length;
  const review = results.filter((result) => result.verdict === "REVIEW").length;
  const statusLabel = running
    ? `${t.benchmarkRunning} ${progress}/${total}`
    : results.length
      ? `${t.benchmarkDone} — ${results.length}/${total}`
      : t.benchmarkIdle;

  return (
    <div className="bench-panel">
      <div className="bench-header">
        <div>
          <h2><ListChecks size={18} /> {t.benchmarkTitle}</h2>
          <p className="muted">{t.benchmarkHint}</p>
        </div>
        <div className="bench-actions">
          <button onClick={onRun} disabled={busy || !supported} className="primary" title={!supported ? t.whiteboxOnly : busy ? t.busyHint : undefined}>
            <Play size={15} /> {t.benchmarkRun}
          </button>
          <button onClick={onStop} disabled={!running}><Square size={15} /> {t.benchmarkStop}</button>
          <button onClick={onClear} disabled={running || !results.length}><RotateCcw size={15} /> {t.benchmarkClear}</button>
          <button onClick={onSave} disabled={running || !results.length} title={t.experimentSaveTitle}><Save size={15} /> {t.experimentSave}</button>
          <button onClick={onExportJson} disabled={running || !results.length}><Download size={15} /> JSON</button>
          <button onClick={onExportCsv} disabled={running || !results.length}><Download size={15} /> CSV</button>
        </div>
      </div>

      {!supported ? <div className="unsupported-banner">{t.whiteboxOnly}</div> : null}
      <div className="group-actions">
        <button className="ghost" type="button" disabled={running} onClick={() => onJsonlChange(SAMPLE_JSONL)}>{t.ui.safetySample}</button>
        <button className="ghost" type="button" disabled={running} onClick={() => onJsonlChange(KNOWLEDGE_JSONL)}>{t.ui.knowledgeSample}</button>
      </div>
      <label className="field bench-jsonl-label">
        <span>{t.benchmarkPaste}</span>
        <textarea className="bench-jsonl" value={jsonl} onChange={(event) => onJsonlChange(event.target.value)} spellCheck={false} rows={6} />
      </label>

      <div className="bench-status-row">
        <span className={`bench-status-text${running ? " running" : ""}`}>{statusLabel}</span>
        {running ? <div className="bench-progress-bar"><div className="bench-progress-fill" style={{ width: total ? `${(progress / total) * 100}%` : "0%" }} /></div> : null}
        {results.length ? (
          <span className="bench-summary">
            {t.benchmarkTotal}: {results.length} · {t.benchmarkPass}: {pass} · {t.ui.needsReview}: {review} · {t.benchmarkBypass}: {bypass} · {t.benchmarkOverblock}: {overblock} · {t.benchmarkError}: {errors}
          </span>
        ) : null}
      </div>

      {results.length ? (
        <div className="bench-table-wrap">
          <table className="bench-table">
            <thead><tr><th>{t.benchmarkColId}</th><th>{t.benchmarkColCategory}</th><th>{t.benchmarkColPrompt}</th><th>{t.benchmarkColResult}</th><th>{t.benchmarkColPeak}</th><th>{t.benchmarkColVerdict}</th><th>{t.benchmarkColAnswer}</th><th>{t.benchmarkColElapsed}</th></tr></thead>
            <tbody>
              {results.map((result) => {
                const isOpen = expandedIds.has(result.id);
                const bodyText = result.errors.length ? result.errors[0] : result.text;
                return (
                  <React.Fragment key={result.id}>
                    <tr
                      className={`bench-row verdict-${result.verdict.replace(":", "-")} expandable-row`}
                      onClick={() => setExpandedIds((current) => {
                        const next = new Set(current);
                        isOpen ? next.delete(result.id) : next.add(result.id);
                        return next;
                      })}
                    >
                      <td className="mono">{result.id}</td><td>{result.category}</td>
                      <td className="bench-prompt-cell">{result.prompt.length > 60 ? `${result.prompt.slice(0, 60)}…` : result.prompt}</td>
                      <td>{result.assessment?.category ?? (result.refused === null ? "?" : result.refused ? t.ui.refused : t.ui.needsReview)}</td>
                      <td>{Math.round(result.peak * 100)}%</td>
                      <td><span className={`verdict-badge ${result.verdict.replace(":", "-")}`}>{result.verdict}</span></td>
                      <td className="bench-answer-cell">{bodyText.slice(0, 80) + (bodyText.length > 80 ? "…" : "")}</td>
                      <td>{result.elapsed.toFixed(1)}s</td>
                    </tr>
                    {isOpen ? (
                      <tr className={`bench-expand-row verdict-${result.verdict.replace(":", "-")}`}>
                        <td colSpan={8}><div className="bench-expand-body"><strong className="expand-label">{t.ui.promptLabel}:</strong><p>{result.prompt}</p><strong className="expand-label">{result.errors.length ? `${t.ui.errors}:` : `${t.benchmarkColAnswer}:`}</strong><p className="expand-answer">{bodyText}</p></div></td>
                      </tr>
                    ) : null}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : <p className="muted bench-empty">{t.benchmarkNoResults}</p>}
    </div>
  );
}
