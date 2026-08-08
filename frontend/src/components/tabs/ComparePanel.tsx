import React, { useState } from "react";
import { Download, Play, Save, Square, Swords } from "lucide-react";
import { JAILBREAK_MODES, jailbreakModeLabel, parseNumberList } from "../../app/benchmarking";
import type { Translation } from "../../i18n";
import type { CompareResult, ModelInfo } from "../../types";

type CompareKind = "modes" | "models" | "sweep";

interface ComparePanelProps {
  t: Translation;
  prompt: string;
  onPromptChange: (value: string) => void;
  kind: CompareKind;
  onKindChange: (value: CompareKind) => void;
  secondModel: string;
  onSecondModelChange: (value: string) => void;
  modelOptions: ModelInfo[];
  sweepStrengths: string;
  onSweepStrengthsChange: (value: string) => void;
  results: CompareResult[];
  running: boolean;
  onRun: () => void;
  onStop: () => void;
  onSave: () => void;
  onExportJson: () => void;
  onExportCsv: () => void;
  supported: boolean;
  busy: boolean;
}

export function ComparePanel({
  t, prompt, onPromptChange, kind, onKindChange, secondModel, onSecondModelChange,
  modelOptions, sweepStrengths, onSweepStrengthsChange, results, running, onRun,
  onStop, onSave, onExportJson, onExportCsv, supported, busy
}: ComparePanelProps) {
  const [expandedModes, setExpandedModes] = useState<Set<string>>(new Set());
  const sweepCount = parseNumberList(sweepStrengths).length;

  return (
    <div className="compare-panel">
      <div className="bench-header">
        <div>
          <h2><Swords size={18} /> {t.compareTitle}</h2>
          <p className="muted">{kind === "models" ? t.ui.modelsHint : kind === "sweep" ? t.ui.sweepHint : t.ui.modesHint}</p>
        </div>
        <div className="bench-actions">
          <button onClick={onRun} disabled={busy || !prompt.trim() || !supported || (kind === "models" && !secondModel) || (kind === "sweep" && !sweepCount)} className="primary" title={!supported ? t.whiteboxOnly : busy ? t.busyHint : undefined}>
            <Play size={15} /> {kind === "models" ? t.ui.runBothModels : kind === "sweep" ? t.ui.runSweep : t.compareRun}
          </button>
          <button onClick={onStop} disabled={!running}><Square size={15} /> {t.compareStop}</button>
          <button onClick={onSave} disabled={running || !results.length} title={t.experimentSaveTitle}><Save size={15} /> {t.experimentSave}</button>
          <button onClick={onExportJson} disabled={running || !results.length}><Download size={15} /> JSON</button>
          <button onClick={onExportCsv} disabled={running || !results.length}><Download size={15} /> CSV</button>
        </div>
      </div>

      {!supported ? <div className="unsupported-banner">{t.whiteboxOnly}</div> : null}
      <div className="field-grid">
        <label className="field">
          <span>{t.ui.comparisonType}</span>
          <select value={kind} onChange={(event) => onKindChange(event.target.value as CompareKind)}>
            <option value="modes">{t.ui.modesOption}</option>
            <option value="models">{t.ui.modelsOption}</option>
            <option value="sweep">{t.ui.sweepOption}</option>
          </select>
        </label>
        {kind === "models" ? (
          <label className="field">
            <span>{t.ui.secondModel}</span>
            <select value={secondModel} onChange={(event) => onSecondModelChange(event.target.value)}>
              <option value="">{t.ui.chooseModel}</option>
              {modelOptions.map((item) => <option key={`${item.adapter}-${item.id}`} value={item.id}>{item.label}</option>)}
            </select>
          </label>
        ) : null}
        {kind === "sweep" ? (
          <label className="field"><span>{t.ui.strengthValues}</span><input value={sweepStrengths} onChange={(event) => onSweepStrengthsChange(event.target.value)} placeholder="0.5, 0.75, 1.0, 1.25" /></label>
        ) : null}
      </div>

      <label className="field">
        <span>{t.comparePromptLabel}</span>
        <textarea className="chat-input" value={prompt} onChange={(event) => onPromptChange(event.target.value)} rows={3} spellCheck={false} placeholder={t.prompt} />
      </label>
      {running ? <p className="bench-status-text running">{results.length}/{kind === "models" ? 2 : kind === "sweep" ? sweepCount : JAILBREAK_MODES.length + 1}…</p> : null}

      {results.length ? (
        <div className="bench-table-wrap">
          <table className="bench-table">
            <thead><tr><th>{t.compareColMode}</th><th>{t.compareColPeak}</th><th>{t.compareColState}</th><th>{t.compareColResult}</th><th>{t.compareColAnswer}</th><th>{t.compareColElapsed}</th></tr></thead>
            <tbody>
              {results.map((result) => {
                const isOpen = expandedModes.has(result.mode);
                const verifiedCompliance = result.assessment?.category === "complete_compliance" && result.assessment.coherent && !result.assessment.truncated;
                const rowClass = result.refused === null ? "bench-row" : result.refused ? "bench-row verdict-PASS" : verifiedCompliance ? "bench-row verdict-FAIL-bypass" : "bench-row";
                return (
                  <React.Fragment key={result.mode}>
                    <tr
                      className={`${rowClass} expandable-row`}
                      onClick={() => setExpandedModes((current) => {
                        const next = new Set(current);
                        isOpen ? next.delete(result.mode) : next.add(result.mode);
                        return next;
                      })}
                    >
                      <td><strong>{result.mode === "baseline" ? t.compareBaseline : jailbreakModeLabel(result.mode, t)}</strong></td>
                      <td>{Math.round(result.peak * 100)}%</td><td>{result.state}</td>
                      <td>{result.assessment?.category ?? (result.refused === null ? "?" : result.refused ? t.ui.refused : t.ui.needsReview)}</td>
                      <td className="bench-answer-cell">{result.text.slice(0, 100) + (result.text.length > 100 ? "…" : "")}</td>
                      <td>{result.elapsed.toFixed(1)}s</td>
                    </tr>
                    {isOpen ? (
                      <tr className={`bench-expand-row ${rowClass.replace("bench-row", "").trim()}`}>
                        <td colSpan={6}><div className="bench-expand-body"><strong className="expand-label">{t.compareColAnswer}:</strong><p className="expand-answer">{result.text || `(${t.ui.empty})`}</p>{result.errors.length ? <><strong className="expand-label">{t.ui.errors}:</strong><p className="expand-answer">{result.errors.join("\n")}</p></> : null}</div></td>
                      </tr>
                    ) : null}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : !running ? <p className="muted bench-empty">{t.compareNoResults}</p> : null}
    </div>
  );
}
