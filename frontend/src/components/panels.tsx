/** Stateless telemetry views. Every one of these takes plain data and renders
 *  it — no fetching, no run state — so they can be dropped into any page. */

import React from "react";
import { safetyNote, safetyStateLabel, translations, type Language, type Translation } from "../i18n";
import type { LayerOp } from "../interventions";
import type {
  AttentionTrace,
  BlackBoxMetrics,
  Candidate,
  HeadMap,
  LayerMetric,
  LensToken,
  SafetyTrace,
  ThinkPhaseSummary
} from "../types";

export function PanelTitle({ icon, title, aside }: { icon: React.ReactNode; title: string; aside?: React.ReactNode }) {
  return (
    <div className="panel-title">
      {icon}
      <h2>{title}</h2>
      {aside ? <div className="panel-title-aside">{aside}</div> : null}
    </div>
  );
}

export function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

const OP_MARK: Record<LayerOp["action"], string> = { mute: "✕", scale: "↓", boost: "↑" };

export function LayerGrid({
  layers,
  layerCount,
  activityLabel,
  safetyLabel,
  uncertaintyLabel,
  ops,
  onToggle,
  opHint
}: {
  layers: LayerMetric[];
  layerCount: number;
  activityLabel: string;
  safetyLabel: string;
  uncertaintyLabel: string;
  /** Layer index → the intervention applied to it. */
  ops?: Record<number, LayerOp>;
  /** Click-to-intervene, mirroring the head map. Omit for a read-only grid. */
  onToggle?: (layer: number) => void;
  opHint?: string;
}) {
  const source = layers.length ? layers : Array.from({ length: layerCount }, (_, layer) => ({ layer, activity: 0, safety: 0, uncertainty: 0 }));
  const interactive = typeof onToggle === "function";

  return (
    <div className="layer-grid">
      {source.map((item) => {
        const op = ops?.[item.layer];
        const title =
          `L${item.layer}\n${activityLabel}: ${Math.round(item.activity * 100)}%\n` +
          `${safetyLabel}: ${Math.round(item.safety * 100)}%\n${uncertaintyLabel}: ${Math.round(item.uncertainty * 100)}%` +
          (op ? `\n→ ${op.action}${op.action === "mute" ? "" : ` ×${op.scale}`}` : interactive && opHint ? `\n${opHint}` : "");

        const content = (
          <>
            <span className="layer-label">L{item.layer}</span>
            <div className="layer-indicators">
              {item.safety > 0.01 && <span className="ind-red">{Math.round(item.safety * 100)}%</span>}
              {item.uncertainty > 0.01 && <span className="ind-yellow">{Math.round(item.uncertainty * 100)}%</span>}
            </div>
            {op ? <span className={`layer-op ${op.action}`}>{OP_MARK[op.action]}</span> : null}
          </>
        );

        const style = {
          "--activity": item.activity,
          "--safety": item.safety,
          "--uncertainty": item.uncertainty
        } as React.CSSProperties;

        const className = `layer-cell${op ? ` op-${op.action}` : ""}${interactive ? " interactive" : ""}`;

        return interactive ? (
          <button type="button" className={className} key={item.layer} title={title} style={style} onClick={() => onToggle!(item.layer)}>
            {content}
          </button>
        ) : (
          <div className={className} key={item.layer} title={title} style={style}>
            {content}
          </div>
        );
      })}
    </div>
  );
}

export function SafetyView({ safety, language }: { safety: SafetyTrace | null; language: Language }) {
  const t = translations[language];
  if (!safety) return <p className="muted">{t.noSafetyTrace}</p>;
  return (
    <div className="safety-view">
      <div className="score-line">
        <span>{t.state}</span>
        <strong>{safetyStateLabel(language, safety.state)}</strong>
      </div>
      <div className="progress">
        <span style={{ width: `${Math.round(safety.score * 100)}%` }} />
      </div>
      <div className="safety-grid">
        <Metric label={t.firstTrigger} value={safety.first_trigger_layer === null ? "-" : `L${safety.first_trigger_layer}`} />
        <Metric label={t.locked} value={safety.locked_layer === null ? "-" : `L${safety.locked_layer}`} />
      </div>
      <p>{safetyNote(language, safety.state, safety.notes)}</p>
    </div>
  );
}

export function BarList({ items, emptyLabel }: { items: Array<{ label: string; value: number }>; emptyLabel: string }) {
  if (!items.length) return <p className="muted">{emptyLabel}</p>;
  return (
    <div className="bar-list">
      {items.map((item, index) => (
        <div className="bar-row" key={`${item.label}-${index}`}>
          <span>{item.label}</span>
          <div className="bar-track">
            <i style={{ width: `${Math.round(item.value * 100)}%` }} />
          </div>
          <strong>{Math.round(item.value * 100)}%</strong>
        </div>
      ))}
    </div>
  );
}

export function TopKList({ items, emptyLabel, spaceLabel }: { items: Candidate[]; emptyLabel: string; spaceLabel: string }) {
  if (!items.length) return <p className="muted">{emptyLabel}</p>;
  return <BarList items={items.map((item) => ({ label: item.token || spaceLabel, value: item.prob }))} emptyLabel={emptyLabel} />;
}

export function LayerLensView({ items, emptyLabel }: { items: LensToken[]; emptyLabel: string }) {
  if (!items.length) return <p className="muted">{emptyLabel}</p>;
  return (
    <div className="lens-flow">
      {items.map((item, index) => {
        const changed = index === 0 || items[index - 1].token !== item.token;
        const label = item.token.trim() || "␣";
        return (
          <span
            className={`lens-chip${changed ? " changed" : ""}`}
            key={item.layer}
            title={`L${item.layer} · ${Math.round(item.prob * 100)}%`}
          >
            <em>L{item.layer}</em>
            <b style={{ opacity: 0.4 + Math.min(item.prob, 1) * 0.6 }}>{label}</b>
          </span>
        );
      })}
    </div>
  );
}

export function HeadMapView({
  map,
  muted,
  onToggle,
  emptyLabel,
  mutedLabel
}: {
  map: HeadMap | null;
  muted: Set<string>;
  onToggle: (layer: number, head: number) => void;
  emptyLabel: string;
  mutedLabel: string;
}) {
  if (!map || !map.layers.length) return <p className="muted">{emptyLabel}</p>;
  return (
    <div className="headmap">
      <div className="headmap-grid" style={{ "--heads": map.n_heads } as React.CSSProperties}>
        {map.layers.map((layer) => (
          <div className="headmap-row" key={layer.layer}>
            <span className="headmap-label">L{layer.layer}</span>
            <div className="headmap-cells">
              {layer.heads.map((head) => {
                const key = `${layer.layer}:${head.head}`;
                const isMuted = muted.has(key);
                return (
                  <button
                    type="button"
                    key={head.head}
                    className={`head-cell${isMuted ? " muted" : ""}`}
                    style={{ "--score": head.score } as React.CSSProperties}
                    title={`L${layer.layer} H${head.head} · refusal ${Math.round(head.score * 100)}%${isMuted ? " · MUTE" : ""}`}
                    onClick={() => onToggle(layer.layer, head.head)}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
      {muted.size > 0 ? (
        <p className="headmap-foot">
          {muted.size} {mutedLabel}
        </p>
      ) : null}
    </div>
  );
}

export function AttentionView({ trace, emptyLabel }: { trace: AttentionTrace | null; emptyLabel: string }) {
  if (!trace || !trace.tokens.length) return <p className="muted">{emptyLabel}</p>;
  return (
    <div className="attention-list">
      {trace.tokens.map((token, index) => (
        <span key={`${token}-${index}`} style={{ opacity: 0.32 + Math.min((trace.weights[index] ?? 0) * 6, 0.68) }}>
          {token}
        </span>
      ))}
    </div>
  );
}

export function ThinkPhaseView({ summary, currentPhase, t }: { summary: ThinkPhaseSummary | null; currentPhase: "think" | "answer"; t: Translation }) {
  if (!summary) {
    return <p className="muted">{t.thinkPhaseNone}</p>;
  }

  const maxDelta = Math.max(...summary.delta.map(Math.abs), 0.001);

  return (
    <div className="think-phase-view">
      <div className="think-phase-indicator">
        <span className={`phase-badge ${currentPhase}`}>{t.phaseLabel}: {currentPhase === "think" ? t.thinkSteps : t.answerSteps}</span>
        <span className="think-step-count">{summary.think_steps} {t.thinkSteps} / {summary.answer_steps} {t.answerSteps}</span>
      </div>
      <div className="think-dominant">
        <strong>{t.thinkDominant}:</strong>{" "}
        {summary.dominant_think_layers.map((l) => `L${l}`).join(", ")}
      </div>
      <div className="think-delta-grid">
        <span className="think-delta-label">{t.thinkDelta}</span>
        <div className="think-delta-bars">
          {summary.delta.map((d, i) => {
            const pct = (d / maxDelta) * 50;
            const isPositive = d >= 0;
            return (
              <div
                key={i}
                className="think-delta-bar"
                title={`L${i}: ${d > 0 ? "+" : ""}${(d * 100).toFixed(1)}%`}
              >
                <span className="think-layer-num">L{i}</span>
                <div className="think-bar-track">
                  {isPositive ? (
                    <div className="think-bar-fill positive" style={{ width: `${Math.abs(pct)}%`, marginLeft: "50%" }} />
                  ) : (
                    <div className="think-bar-fill negative" style={{ width: `${Math.abs(pct)}%`, marginLeft: `${50 - Math.abs(pct)}%` }} />
                  )}
                  <div className="think-bar-center" />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function RuntimeView({ metrics, log, language }: { metrics: BlackBoxMetrics | null; log: string[]; language: Language }) {
  const t = translations[language];
  return (
    <div className="runtime-view">
      {metrics ? (
        <div className="runtime-grid">
          <Metric label={t.promptTokens} value={String(metrics.prompt_eval_count ?? "-")} />
          <Metric label={t.outputTokens} value={String(metrics.eval_count ?? "-")} />
        </div>
      ) : (
        <p className="muted">{t.noRuntime}</p>
      )}
      <div className="log-list">
        {log.length ? log.map((item, index) => <span key={`${item}-${index}`}>{item}</span>) : <span className="muted">{t.noRuntime}</span>}
      </div>
    </div>
  );
}
