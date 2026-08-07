/** Side-by-side comparison of two saved experiments.
 *
 *  Two forms, picked by the data's job (see the dataviz method):
 *    - per-layer delta   → polarity, so a DIVERGING column chart (two hues +
 *                          neutral midpoint), centred on zero
 *    - safety over time  → two distinct series, so a multi-line chart with a
 *                          legend (identity is never color-alone)
 *
 *  Every chart has a table view behind a toggle, so no value is gated on color.
 */

import React, { useMemo, useState } from "react";
import { biggestMovers, describeRun, type ExperimentDiff as Diff, type LayerDelta } from "../diff";
import type { Translation } from "../i18n";

/* Diverging pair + neutral midpoint, from the reference palette's dark steps.
 * Validated against surface #14171b: band/chroma/contrast PASS, CVD ΔE 66.4.
 * Red = more refusal in B, blue = less. */
const MORE_REFUSAL = "#e66767";
const LESS_REFUSAL = "#3987e5";
const NEUTRAL = "#383835";

/* Run identity for the overlay (categorical slots 1 and 2, ΔE 69.8). */
export const RUN_A = "#3987e5";
export const RUN_B = "#199e70";

const AXIS = "#6d7681";
const GRID = "#23282f";
const SURFACE = "#14171b";

/** Column with its data-end rounded and its baseline end square. */
function columnPath(x: number, width: number, zeroY: number, valueY: number, radius = 4): string {
  const up = valueY < zeroY;
  const height = Math.abs(zeroY - valueY);
  const r = Math.min(radius, width / 2, height);
  if (height < 0.5) return "";
  return up
    ? `M${x},${zeroY} V${valueY + r} Q${x},${valueY} ${x + r},${valueY} H${x + width - r} Q${x + width},${valueY} ${x + width},${valueY + r} V${zeroY} Z`
    : `M${x},${zeroY} V${valueY - r} Q${x},${valueY} ${x + r},${valueY} H${x + width - r} Q${x + width},${valueY} ${x + width},${valueY - r} V${zeroY} Z`;
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function signed(value: number): string {
  const rounded = Math.round(value * 100);
  return `${rounded > 0 ? "+" : ""}${rounded}%`;
}

/* ── Diverging per-layer delta ────────────────────────────────────────────── */

function LayerDeltaChart({ deltas, t }: { deltas: LayerDelta[]; t: Translation }) {
  const [hover, setHover] = useState<number | null>(null);
  const width = 680;
  const height = 190;
  const padLeft = 34;
  const padY = 14;

  const model = useMemo(() => {
    const max = Math.max(...deltas.map((item) => Math.abs(item.delta)), 0.01);
    const plotW = width - padLeft - 10;
    const band = plotW / Math.max(deltas.length, 1);
    // 2px surface gap between neighbours; cap thickness so wide charts stay thin.
    const barW = Math.max(1, Math.min(24, band - 2));
    const zeroY = height / 2;
    const scale = (height / 2 - padY) / max;
    return { max, band, barW, zeroY, scale, plotW };
  }, [deltas]);

  if (!deltas.length) return null;

  return (
    <svg className="delta-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={t.diffLayerChart}>
      <line x1={padLeft} x2={width - 10} y1={padY} y2={padY} stroke={GRID} strokeWidth={1} />
      <line x1={padLeft} x2={width - 10} y1={height - padY} y2={height - padY} stroke={GRID} strokeWidth={1} />
      {/* Neutral midpoint — the diverging zero line. */}
      <line x1={padLeft} x2={width - 10} y1={model.zeroY} y2={model.zeroY} stroke={NEUTRAL} strokeWidth={1} />

      <text x={4} y={padY + 4} fill={AXIS} fontSize={9}>{signed(model.max)}</text>
      <text x={4} y={model.zeroY + 3} fill={AXIS} fontSize={9}>0</text>
      <text x={4} y={height - padY + 4} fill={AXIS} fontSize={9}>{signed(-model.max)}</text>

      {deltas.map((item, index) => {
        const x = padLeft + index * model.band + (model.band - model.barW) / 2;
        const valueY = model.zeroY - item.delta * model.scale;
        const path = columnPath(x, model.barW, model.zeroY, valueY);
        return (
          <g key={item.layer} onMouseEnter={() => setHover(index)} onMouseLeave={() => setHover(null)}>
            {/* Full-height hit target: the bars themselves are too thin to hover. */}
            <rect x={padLeft + index * model.band} y={padY} width={model.band} height={height - padY * 2} fill="transparent" />
            {path ? <path d={path} fill={item.delta >= 0 ? MORE_REFUSAL : LESS_REFUSAL} /> : null}
          </g>
        );
      })}

      {hover !== null && deltas[hover] ? (
        <g>
          <text
            x={Math.min(width - 10, Math.max(padLeft, padLeft + hover * model.band + model.band / 2))}
            y={deltas[hover].delta >= 0 ? padY - 3 : height - 3}
            fill="#e6e9ec"
            fontSize={10}
            textAnchor="middle"
          >
            L{deltas[hover].layer} {signed(deltas[hover].delta)}
          </text>
        </g>
      ) : null}
    </svg>
  );
}

/* ── Safety over tokens, both runs ────────────────────────────────────────── */

function TimelineOverlay({ diff, t }: { diff: Diff; t: Translation }) {
  const width = 680;
  const height = 130;
  const padTop = 10;
  const padBottom = 18;
  const padLeft = 34;

  const series = useMemo(() => {
    const build = (steps: Array<{ safety: number }> | undefined) => steps?.map((step) => step.safety) ?? [];
    return { a: build(diff.timelineA?.steps), b: build(diff.timelineB?.steps) };
  }, [diff]);

  const maxLen = Math.max(series.a.length, series.b.length);
  if (!maxLen) return <p className="muted">{t.diffNoTimeline}</p>;

  const x = (index: number) => padLeft + (maxLen === 1 ? 0 : (index / (maxLen - 1)) * (width - padLeft - 10));
  const y = (value: number) => padTop + (1 - Math.min(value, 1)) * (height - padTop - padBottom);
  const points = (values: number[]) => values.map((value, index) => `${x(index)},${y(value)}`).join(" ");

  return (
    <>
      {/* Legend: two series, so identity never rests on color alone. */}
      <div className="chart-legend">
        <span><i style={{ background: RUN_A }} />A · {describeRun(diff.a)}</span>
        <span><i style={{ background: RUN_B }} />B · {describeRun(diff.b)}</span>
      </div>
      <svg className="overlay-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={t.diffTimelineChart}>
        {[0, 0.5, 1].map((fraction) => {
          const gy = padTop + fraction * (height - padTop - padBottom);
          return <line key={fraction} x1={padLeft} x2={width - 10} y1={gy} y2={gy} stroke={GRID} strokeWidth={1} />;
        })}
        <text x={4} y={padTop + 4} fill={AXIS} fontSize={9}>100%</text>
        <text x={4} y={height - padBottom} fill={AXIS} fontSize={9}>0</text>

        {series.a.length ? (
          <polyline points={points(series.a)} fill="none" stroke={RUN_A} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        ) : null}
        {series.b.length ? (
          <polyline points={points(series.b)} fill="none" stroke={RUN_B} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        ) : null}

        {series.a.length ? (
          <circle cx={x(series.a.length - 1)} cy={y(series.a[series.a.length - 1])} r={4} fill={RUN_A} stroke={SURFACE} strokeWidth={2} />
        ) : null}
        {series.b.length ? (
          <circle cx={x(series.b.length - 1)} cy={y(series.b[series.b.length - 1])} r={4} fill={RUN_B} stroke={SURFACE} strokeWidth={2} />
        ) : null}

        <text x={padLeft} y={height - 4} fill={AXIS} fontSize={9}>0</text>
        <text x={width - 10} y={height - 4} fill={AXIS} fontSize={9} textAnchor="end">{t.timelineToken} {maxLen - 1}</text>
      </svg>
    </>
  );
}

/* ── Panel ────────────────────────────────────────────────────────────────── */

export function ExperimentDiffView({ diff, t, onBack }: { diff: Diff; t: Translation; onBack: () => void }) {
  const [showTable, setShowTable] = useState(false);
  const movers = useMemo(() => biggestMovers(diff.safetyByLayer), [diff]);

  const scalarLabel: Record<string, string> = {
    refused: t.diffRefused,
    safety: t.safety,
    peakSafety: t.diffPeakSafety,
    outputTokens: t.outputTokens,
    promptTokens: t.promptTokens,
  };

  const renderValue = (value: number | boolean | null, key: string) => {
    if (value === null) return "—";
    if (typeof value === "boolean") return value ? t.diffYes : t.diffNo;
    return key.toLowerCase().includes("token") ? String(value) : pct(value);
  };

  return (
    <div className="diff-panel">
      <div className="bench-header">
        <div>
          <h2>{t.diffTitle}</h2>
          <p className="muted">{t.diffHint}</p>
        </div>
        <div className="bench-actions">
          <button className="ghost" onClick={() => setShowTable((value) => !value)}>
            {showTable ? t.diffShowChart : t.diffShowTable}
          </button>
          <button className="ghost" onClick={onBack}>{t.diffBack}</button>
        </div>
      </div>

      {!diff.samePrompt ? <div className="unsupported-banner">{t.diffDifferentPrompts}</div> : null}

      <div className="diff-sides">
        <div className="diff-side">
          <span className="diff-tag" style={{ background: RUN_A }}>A</span>
          <div>
            <strong>{diff.a.label || diff.a.id}</strong>
            <span className="muted">{describeRun(diff.a)} · {String(diff.a.config?.model ?? "").split(/[\\/]/).pop()}</span>
          </div>
        </div>
        <div className="diff-side">
          <span className="diff-tag" style={{ background: RUN_B }}>B</span>
          <div>
            <strong>{diff.b.label || diff.b.id}</strong>
            <span className="muted">{describeRun(diff.b)} · {String(diff.b.config?.model ?? "").split(/[\\/]/).pop()}</span>
          </div>
        </div>
      </div>

      <table className="diff-scalars">
        <thead>
          <tr>
            <th>{t.diffMetric}</th><th>A</th><th>B</th><th>Δ</th>
          </tr>
        </thead>
        <tbody>
          {diff.scalars.map((item) => (
            <tr key={item.key}>
              <td>{scalarLabel[item.key] ?? item.key}</td>
              <td>{renderValue(item.a, item.key)}</td>
              <td>{renderValue(item.b, item.key)}</td>
              <td className={item.delta === null ? "" : item.delta > 0 ? "delta-up" : item.delta < 0 ? "delta-down" : ""}>
                {item.delta === null
                  ? "—"
                  : item.key.toLowerCase().includes("token")
                  ? `${item.delta > 0 ? "+" : ""}${item.delta}`
                  : signed(item.delta)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <section className="card">
        <div className="chart-head">
          <h3>{t.diffLayerChart}</h3>
          <span className="chart-legend-note">
            <i style={{ background: LESS_REFUSAL }} className="key-dot" /> {t.diffLess}
            <i style={{ background: MORE_REFUSAL }} className="key-dot" /> {t.diffMore}
          </span>
        </div>
        {!diff.layersComparable ? (
          <p className="muted">{t.diffLayersIncomparable}</p>
        ) : showTable ? (
          <div className="bench-table-wrap">
            <table className="bench-table">
              <thead><tr><th>{t.layer}</th><th>A</th><th>B</th><th>Δ</th></tr></thead>
              <tbody>
                {diff.safetyByLayer.map((item) => (
                  <tr key={item.layer}>
                    <td className="mono">L{item.layer}</td>
                    <td>{pct(item.a)}</td>
                    <td>{pct(item.b)}</td>
                    <td className={item.delta > 0 ? "delta-up" : item.delta < 0 ? "delta-down" : ""}>{signed(item.delta)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <>
            <LayerDeltaChart deltas={diff.safetyByLayer} t={t} />
            <p className="chart-foot-note">
              {t.diffBiggestMovers}: {movers.map((item) => `L${item.layer} ${signed(item.delta)}`).join(" · ")}
            </p>
          </>
        )}
      </section>

      <section className="card">
        <div className="chart-head"><h3>{t.diffTimelineChart}</h3></div>
        <TimelineOverlay diff={diff} t={t} />
      </section>

      {diff.concepts.length ? (
        <section className="card">
          <div className="chart-head">
            <h3>{t.diffConcepts}</h3>
            <span className="chart-legend-note">
              <i style={{ background: LESS_REFUSAL }} className="key-dot" /> A
              <i style={{ background: MORE_REFUSAL }} className="key-dot" /> B
            </span>
          </div>
          <p className="chart-foot-note">{t.diffConceptsHint}</p>
          <ul className="concept-diff">
            {diff.concepts.map((item) => (
              <li key={item.name}>
                <span className="concept-name">{t.conceptsMap[item.name as keyof typeof t.conceptsMap] ?? item.name}</span>
                {/* Paired bars: the shift in magnitude and in peak layer, side by side. */}
                <span className="concept-diff-bars">
                  <i style={{ width: `${Math.round(item.a * 100)}%`, background: LESS_REFUSAL }} />
                  <i style={{ width: `${Math.round(item.b * 100)}%`, background: MORE_REFUSAL }} />
                </span>
                <span className="concept-diff-nums">
                  {pct(item.a)} → {pct(item.b)}
                </span>
                <span className={`concept-diff-delta${item.delta > 0 ? " delta-up" : item.delta < 0 ? " delta-down" : ""}`}>
                  {signed(item.delta)}
                </span>
                <span className="concept-diff-layer">
                  {item.layerA !== null ? `L${item.layerA}` : "—"}
                  {item.layerB !== null && item.layerB !== item.layerA ? ` → L${item.layerB}` : ""}
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="card">
        <div className="chart-head"><h3>{t.diffOutputs}</h3></div>
        <div className="diff-outputs">
          <div>
            <span className="diff-tag" style={{ background: RUN_A }}>A</span>
            <p>{diff.textA || <em className="muted">{t.waitingOutput}</em>}</p>
          </div>
          <div>
            <span className="diff-tag" style={{ background: RUN_B }}>B</span>
            <p>{diff.textB || <em className="muted">{t.waitingOutput}</em>}</p>
          </div>
        </div>
      </section>
    </div>
  );
}
