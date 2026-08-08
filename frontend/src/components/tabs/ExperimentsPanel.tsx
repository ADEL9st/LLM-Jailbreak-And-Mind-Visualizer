import { useState } from "react";
import { Archive, Download, RotateCcw, Swords, Trash2, Upload } from "lucide-react";
import { API_BASE } from "../../app/runtime";
import type { Translation } from "../../i18n";
import type { ExperimentSummary, ManualVerdict } from "../../types";

interface ExperimentsPanelProps {
  t: Translation;
  items: ExperimentSummary[];
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  onOpen: (id: string) => void;
  onDownload: (id: string) => void;
  onDelete: (id: string) => void;
  onCompare: (idA: string, idB: string) => void;
  onReview: (id: string, verdict: ManualVerdict, notes: string) => void;
}

export function ExperimentsPanel({
  t, items, loading, error, onRefresh, onOpen, onDownload, onDelete, onCompare, onReview
}: ExperimentsPanelProps) {
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [reviewDrafts, setReviewDrafts] = useState<Record<string, { verdict: ManualVerdict; notes: string }>>({});
  const [selected, setSelected] = useState<string[]>([]);

  const toggleSelected = (id: string) => {
    setSelected((current) => current.includes(id)
      ? current.filter((item) => item !== id)
      : [...current, id].slice(-2));
  };

  return (
    <div className="experiments-panel">
      <div className="bench-header">
        <div><h2><Archive size={18} /> {t.experimentsTitle}</h2><p className="muted">{t.experimentsHint}</p></div>
        <div className="bench-actions">
          {selected.length ? <span className="select-hint">{selected.length === 2 ? t.diffReady : t.diffPickSecond}</span> : null}
          <button className="primary" onClick={() => onCompare(selected[0], selected[1])} disabled={selected.length !== 2} title={t.diffCompareTitle}><Swords size={15} /> {t.diffCompare}</button>
          <button onClick={onRefresh} disabled={loading}><RotateCcw size={15} /> {t.experimentsRefresh}</button>
        </div>
      </div>

      {error ? <div className="unsupported-banner">{error}</div> : null}
      {loading && !items.length ? <p className="muted bench-empty">{t.experimentsLoading}</p> : !items.length ? <p className="muted bench-empty">{t.experimentsEmpty}</p> : (
        <div className="experiment-list">
          {items.map((item) => {
            const reviewDraft = reviewDrafts[item.id] ?? { verdict: item.review_verdict ?? "unreviewed", notes: "" };
            return (
              <div className={`experiment-card kind-${item.kind}${selected.includes(item.id) ? " selected" : ""}`} key={item.id}>
                <label className="experiment-pick" title={t.diffSelectTitle}>
                  <input type="checkbox" checked={selected.includes(item.id)} onChange={() => toggleSelected(item.id)} disabled={item.kind !== "run"} />
                  <span>{selected.indexOf(item.id) === 0 ? "A" : selected.indexOf(item.id) === 1 ? "B" : ""}</span>
                </label>
                <div className="experiment-card-main">
                  <div className="experiment-card-head"><span className={`experiment-kind ${item.kind}`}>{t.experimentKind[item.kind as keyof typeof t.experimentKind] ?? item.kind}</span><strong className="experiment-label">{item.label || item.id}</strong></div>
                  <div className="experiment-meta">
                    <span className="mono">{new Date(item.created_at).toLocaleString()}</span>
                    {item.adapter ? <span>{item.adapter}</span> : null}
                    {item.model ? <span className="experiment-model">{item.model.split(/[\\/]/).pop()}</span> : null}
                    {item.jailbreak ? <span className="experiment-flag jb">jailbreak: {item.jailbreak_mode || "default"}</span> : null}
                    {item.safety_score !== null ? <span>{t.safety}: {Math.round(item.safety_score * 100)}%</span> : null}
                    {item.refused !== null ? <span className={`experiment-flag ${item.refused ? "refused" : "not-refused"}`}>{item.refused ? t.ui.refused : t.ui.notRefused}</span> : null}
                    {item.row_count ? <span>{item.row_count} {t.experimentRows}</span> : null}
                    {item.output_category ? <span className="experiment-flag">{item.output_category}</span> : null}
                    <label className="review-picker" onClick={(event) => event.stopPropagation()}>
                      <span>{t.ui.manual}</span>
                      <select value={reviewDraft.verdict} onChange={(event) => setReviewDrafts((current) => ({ ...current, [item.id]: { ...reviewDraft, verdict: event.target.value as ManualVerdict } }))}>
                        <option value="unreviewed">{t.ui.unreviewed}</option><option value="pass">{t.ui.pass}</option><option value="partial">{t.ui.partial}</option><option value="fail">{t.ui.fail}</option><option value="inconclusive">{t.ui.inconclusive}</option>
                      </select>
                      <input value={reviewDraft.notes} onChange={(event) => setReviewDrafts((current) => ({ ...current, [item.id]: { ...reviewDraft, notes: event.target.value } }))} placeholder={t.ui.reviewNote} />
                      <button type="button" onClick={() => onReview(item.id, reviewDraft.verdict, reviewDraft.notes)}>{t.ui.saveReview}</button>
                    </label>
                    <span className="muted">{(item.size_bytes / 1024).toFixed(1)} KB</span>
                  </div>
                </div>
                <div className="experiment-card-actions">
                  <button onClick={() => onOpen(item.id)} title={t.experimentOpenTitle}><Upload size={14} /> {t.experimentOpen}</button>
                  <button onClick={() => onDownload(item.id)}><Download size={14} /> JSON</button>
                  {item.row_count ? <a className="experiment-csv-link" href={`${API_BASE}/experiments/${encodeURIComponent(item.id)}/csv`} download={`${item.id}.csv`}><Download size={14} /> CSV</a> : null}
                  {confirmId === item.id ? (
                    <button className="danger" onClick={() => { onDelete(item.id); setConfirmId(null); }} onBlur={() => setConfirmId(null)}><Trash2 size={14} /> {t.experimentConfirmDelete}</button>
                  ) : <button onClick={() => setConfirmId(item.id)} title={t.experimentDeleteTitle}><Trash2 size={14} /></button>}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
