/** Time-axis views of a run.
 *
 *  The backend already streams a full telemetry frame per generated token; until
 *  now the UI overwrote it each step and only ever showed the last frame. These
 *  two views put the token axis back:
 *
 *    - SafetyHeatmap — token × layer refusal projection (sequential, one hue)
 *    - TokenLine     — one measure over tokens (single series, so no legend)
 *
 *  Colors come from the validated palette in `chartPalette` below; do not
 *  hand-pick new ones here.
 */

import React, { useEffect, useMemo, useRef, useState } from "react";
import type { TimelineTrace } from "../types";

/* Validated against the card surface #14171b in dark mode
 * (scripts/validate_palette.js): the two series pass band, chroma, contrast and
 * CVD (worst all-pairs ΔE 66.4). The safety ramp is *sequential*, so it is
 * checked for monotone lightness and single hue (15° spread) — its darkest step
 * deliberately recedes into the surface, which is what "near zero" should look
 * like. */
export const SERIES_SAFETY = "#e66767";
export const SERIES_ENTROPY = "#3987e5";

const SAFETY_RAMP = ["#1e1518", "#3d1b20", "#5e2429", "#822f33", "#a83f41", "#cc5254", "#e66767"];

const AXIS = "#6d7681";
const GRID = "#23282f";

function rampColor(value: number): string {
  const clamped = Math.max(0, Math.min(1, value));
  const index = Math.min(SAFETY_RAMP.length - 1, Math.round(clamped * (SAFETY_RAMP.length - 1)));
  return SAFETY_RAMP[index];
}

function tokenLabel(token: string): string {
  const trimmed = token.replace(/\n/g, "⏎");
  return trimmed.trim() ? trimmed : "␣";
}

/* ── Heatmap ─────────────────────────────────────────────────────────────── */

export function SafetyHeatmap({
  trace,
  title,
  emptyLabel,
  layerLabel,
  tokenLabel: tokenAxisLabel
}: {
  trace: TimelineTrace | null;
  title: string;
  emptyLabel: string;
  layerLabel: string;
  tokenLabel: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; step: number; layer: number; value: number } | null>(null);

  const rows = trace?.layerCount ?? 0;
  const cols = trace?.safetyMatrix.length ?? 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !trace || !rows || !cols) return;

    // Draw at device resolution so a 1px cell doesn't blur on a HiDPI screen.
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const cellW = width / cols;
    const cellH = height / rows;
    for (let step = 0; step < cols; step += 1) {
      const column = trace.safetyMatrix[step];
      for (let layer = 0; layer < rows; layer += 1) {
        ctx.fillStyle = rampColor(column[layer] ?? 0);
        // Math.ceil so sub-pixel columns tile without hairline seams.
        ctx.fillRect(step * cellW, layer * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
  }, [trace, rows, cols]);

  if (!trace || !cols || !rows) return <p className="muted">{emptyLabel}</p>;

  const onMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const step = Math.min(cols - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * cols)));
    const layer = Math.min(rows - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * rows)));
    setHover({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
      step,
      layer,
      value: trace.safetyMatrix[step]?.[layer] ?? 0
    });
  };

  const peak = trace.steps.reduce((best, item) => (item.safety > best.safety ? item : best), trace.steps[0]);

  return (
    <div className="chart">
      <div className="chart-head">
        <h3>{title}</h3>
        <span className="chart-legend-note">
          <i className="ramp-swatch" /> 0 → 100%
        </span>
      </div>
      <div className="heatmap-wrap" onMouseLeave={() => setHover(null)}>
        <div className="heatmap-yaxis">
          <span>L{rows - 1}</span>
          <span>L0</span>
        </div>
        <canvas ref={canvasRef} className="heatmap-canvas" onMouseMove={onMove} />
        {hover ? (
          <div
            className="chart-tip"
            style={{ left: `${hover.x}px`, top: `${hover.y}px` }}
          >
            <strong>L{hover.layer}</strong> · {tokenAxisLabel} {hover.step}
            <br />
            <span className="chart-tip-token">{tokenLabel(trace.steps[hover.step]?.token ?? "")}</span>
            <br />
            {Math.round(hover.value * 100)}%
          </div>
        ) : null}
      </div>
      <div className="chart-foot">
        <span>{layerLabel} × {tokenAxisLabel}</span>
        {peak ? <span>peak {Math.round(peak.safety * 100)}% @ {tokenAxisLabel} {peak.step}</span> : null}
      </div>
    </div>
  );
}

/* ── Line ────────────────────────────────────────────────────────────────── */

export function TokenLine({
  trace,
  metric,
  title,
  color,
  emptyLabel,
  tokenLabel: tokenAxisLabel,
  format
}: {
  trace: TimelineTrace | null;
  metric: "safety" | "entropy";
  title: string;
  color: string;
  emptyLabel: string;
  tokenLabel: string;
  format: (value: number) => string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const width = 640;
  const height = 96;
  const padTop = 8;
  const padBottom = 16;
  const padLeft = 34;

  const model = useMemo(() => {
    if (!trace || !trace.steps.length) return null;
    const values = trace.steps.map((item) => (metric === "safety" ? item.safety : item.entropy));
    const max = Math.max(...values, metric === "safety" ? 1 : 0.5);
    const n = values.length;
    const x = (index: number) => padLeft + (n === 1 ? 0 : (index / (n - 1)) * (width - padLeft - 8));
    const y = (value: number) => padTop + (1 - value / max) * (height - padTop - padBottom);
    return { values, max, n, x, y, points: values.map((value, index) => `${x(index)},${y(value)}`).join(" ") };
  }, [trace, metric]);

  if (!model) return <p className="muted">{emptyLabel}</p>;

  const lastIndex = model.n - 1;
  const active = hover ?? lastIndex;
  const activeValue = model.values[active];

  return (
    <div className="chart">
      <div className="chart-head">
        <h3>{title}</h3>
        <span className="chart-value" style={{ color }}>{format(activeValue)}</span>
      </div>
      <svg
        className="line-chart"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={title}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientX - rect.left) / rect.width;
          const index = Math.round(ratio * width - padLeft) / (width - padLeft - 8);
          setHover(Math.min(lastIndex, Math.max(0, Math.round(index * lastIndex))));
        }}
      >
        {/* Recessive hairline grid — top, middle, baseline only. */}
        {[0, 0.5, 1].map((fraction) => {
          const y = padTop + fraction * (height - padTop - padBottom);
          return <line key={fraction} x1={padLeft} x2={width - 8} y1={y} y2={y} stroke={GRID} strokeWidth={1} />;
        })}
        <text x={4} y={padTop + 4} fill={AXIS} fontSize={9}>{format(model.max)}</text>
        <text x={4} y={height - padBottom} fill={AXIS} fontSize={9}>0</text>

        <polyline points={model.points} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />

        {hover !== null ? (
          <line x1={model.x(active)} x2={model.x(active)} y1={padTop} y2={height - padBottom} stroke={AXIS} strokeWidth={1} />
        ) : null}
        {/* End marker: r=4 (8px) with a 2px surface ring so it stays legible. */}
        <circle cx={model.x(active)} cy={model.y(activeValue)} r={4} fill={color} stroke="#14171b" strokeWidth={2} />

        <text x={padLeft} y={height - 3} fill={AXIS} fontSize={9}>0</text>
        <text x={width - 8} y={height - 3} fill={AXIS} fontSize={9} textAnchor="end">
          {tokenAxisLabel} {lastIndex}
        </text>
      </svg>
      <div className="chart-foot">
        <span>
          {tokenAxisLabel} {active}
          {trace?.steps[active] ? ` · "${tokenLabel(trace.steps[active].token)}"` : ""}
        </span>
      </div>
    </div>
  );
}
