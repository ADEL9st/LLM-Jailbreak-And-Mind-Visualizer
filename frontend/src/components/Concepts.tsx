/** Layer × concept map — "at which layer does the model connect to which
 *  concept", the original idea behind this tool.
 *
 *  Magnitude in a grid, so: heatmap, sequential, one hue (blue — deliberately
 *  not the safety red, so the two heatmaps on the Analysis page never read as
 *  the same measure). Ramp validated for monotone lightness and single hue
 *  (5° spread) against surface #14171b; its darkest step recedes into the
 *  surface because that is what "this layer isn't thinking about this" should
 *  look like.
 */

import React, { useEffect, useRef, useState } from "react";
import type { ConceptTrace } from "../types";

const CONCEPT_RAMP = ["#141c26", "#16324f", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"];

function rampColor(value: number): string {
  const clamped = Math.max(0, Math.min(1, value));
  return CONCEPT_RAMP[Math.min(CONCEPT_RAMP.length - 1, Math.round(clamped * (CONCEPT_RAMP.length - 1)))];
}

export function ConceptMap({
  trace,
  emptyLabel,
  layerLabel,
  peakLabel,
  conceptLabels
}: {
  trace: ConceptTrace | null;
  emptyLabel: string;
  layerLabel: string;
  peakLabel: string;
  /** Localised display name per concept key; falls back to the key. */
  conceptLabels: Record<string, string>;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; layer: number; concept: number; value: number } | null>(null);

  // Defensive: an adapter that still sends the old flat {concepts:[…]} payload
  // has no `names`/`layers`. Render the empty state instead of throwing.
  const rows = trace?.names?.length ?? 0;
  const cols = trace?.layers?.length ?? 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !trace || !rows || !cols) return;
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
    for (let layer = 0; layer < cols; layer += 1) {
      const row = trace.layers[layer];
      for (let concept = 0; concept < rows; concept += 1) {
        ctx.fillStyle = rampColor(row?.[concept] ?? 0);
        ctx.fillRect(layer * cellW, concept * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
  }, [trace, rows, cols]);

  if (!trace || !rows || !cols) return <p className="muted">{emptyLabel}</p>;

  const label = (name: string) => conceptLabels[name] ?? name;

  const onMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const layer = Math.min(cols - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * cols)));
    const concept = Math.min(rows - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * rows)));
    setHover({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
      layer,
      concept,
      value: trace.layers[layer]?.[concept] ?? 0
    });
  };

  return (
    <div className="concept-map">
      <div className="concept-grid" onMouseLeave={() => setHover(null)}>
        <div className="concept-names">
          {trace.names.map((name) => (
            <span key={name} title={name}>{label(name)}</span>
          ))}
        </div>
        <div className="concept-canvas-wrap">
          <canvas ref={canvasRef} className="concept-canvas" onMouseMove={onMove} />
          {hover ? (
            <div className="chart-tip" style={{ left: `${hover.x}px`, top: `${hover.y}px` }}>
              <strong>{label(trace.names[hover.concept])}</strong> · L{hover.layer}
              <br />
              {Math.round(hover.value * 100)}%
            </div>
          ) : null}
          <div className="concept-xaxis">
            <span>L0</span>
            <span>{layerLabel}</span>
            <span>L{cols - 1}</span>
          </div>
        </div>
      </div>

      {/* Ranked readout: the map answers "where", this answers "which". */}
      <ul className="concept-ranked">
        {(trace.concepts ?? []).slice(0, 5).map((item) => (
          <li key={item.name}>
            <span className="concept-swatch" style={{ background: rampColor(item.score) }} />
            <span className="concept-name">{label(item.name)}</span>
            <span className="concept-bar">
              <i style={{ width: `${Math.round(item.score * 100)}%`, background: rampColor(item.score) }} />
            </span>
            <span className="concept-score">{Math.round(item.score * 100)}%</span>
            <span className="concept-peak">{peakLabel} L{item.layer}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
