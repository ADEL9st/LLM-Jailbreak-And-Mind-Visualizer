import { Activity, BrainCircuit, Eye, Grid3x3, RotateCcw, ShieldAlert, Waves } from "lucide-react";
import { ConceptMap } from "../Concepts";
import { SERIES_ENTROPY, SERIES_SAFETY, SafetyHeatmap, TokenLine } from "../Timeline";
import {
  AttentionView,
  HeadMapView,
  LayerGrid,
  LayerLensView,
  PanelTitle,
  RuntimeView,
  SafetyView,
  ThinkPhaseView,
  TopKList
} from "../panels";
import type { LayerOp } from "../../interventions";
import type { Language, Translation } from "../../i18n";
import type {
  AttentionTrace,
  BlackBoxMetrics,
  Candidate,
  ConceptTrace,
  HeadMap,
  LayerMetric,
  LensToken,
  OutputAssessment,
  SafetyTrace,
  ThinkPhaseSummary,
  TimelineTrace
} from "../../types";

interface AnalysisTabProps {
  t: Translation;
  language: Language;
  entropy: number | null;
  hallucinationRisk: number | null;
  dominantLayer: LayerMetric | null;
  safety: SafetyTrace | null;
  promptTokens: number | null;
  outputTokens: number | null;
  outputAssessment: OutputAssessment | null;
  timeline: TimelineTrace | null;
  concepts: ConceptTrace | null;
  conceptLabels: Record<string, string>;
  layerOps: Record<number, LayerOp>;
  onClearLayerOps: () => void;
  brushAction: LayerOp["action"];
  onBrushActionChange: (action: LayerOp["action"]) => void;
  brushScale: number;
  onBrushScaleChange: (scale: number) => void;
  layers: LayerMetric[];
  layerCount: number;
  onToggleLayer: (layer: number) => void;
  lens: LensToken[];
  topK: Candidate[];
  headMap: HeadMap | null;
  mutedHeads: Set<string>;
  onToggleHead: (layer: number, head: number) => void;
  attention: AttentionTrace | null;
  thinkPhase: ThinkPhaseSummary | null;
  currentPhase: "think" | "answer";
  blackBoxMetrics: BlackBoxMetrics | null;
  log: string[];
}

export function AnalysisTab({
  t, language, entropy, hallucinationRisk, dominantLayer, safety, promptTokens,
  outputTokens, outputAssessment, timeline, concepts, conceptLabels, layerOps,
  onClearLayerOps, brushAction, onBrushActionChange, brushScale, onBrushScaleChange,
  layers, layerCount, onToggleLayer, lens, topK, headMap, mutedHeads, onToggleHead,
  attention, thinkPhase, currentPhase, blackBoxMetrics, log
}: AnalysisTabProps) {
  const layerOpCount = Object.keys(layerOps).length;

  return (
    <div className="analysis-page">
      <div className="stat-row">
        <div className="stat-card"><span>{t.entropy}</span><strong>{entropy === null ? "—" : entropy.toFixed(2)}</strong></div>
        <div className="stat-card"><span>{t.hallucination}</span><strong>{hallucinationRisk === null ? "—" : `${Math.round(hallucinationRisk * 100)}%`}</strong></div>
        <div className="stat-card"><span>{t.dominantLayer}</span><strong>{dominantLayer ? `L${dominantLayer.layer}` : "—"}</strong></div>
        <div className="stat-card"><span>{t.safety}</span><strong>{safety ? `${Math.round(safety.score * 100)}%` : "—"}</strong></div>
        <div className="stat-card"><span>{t.totalTokens}</span><strong>{promptTokens !== null && outputTokens !== null ? promptTokens + outputTokens : "—"}</strong></div>
        <div className="stat-card"><span>{t.ui.outputCategory}</span><strong>{outputAssessment?.category ?? "—"}</strong></div>
      </div>

      <div className="card-grid">
        <section className="card span-2">
          <PanelTitle icon={<Activity size={16} />} title={t.timeline} />
          <p className="card-hint">{t.timelineHint}</p>
          <SafetyHeatmap trace={timeline} title={t.timelineHeatmap} emptyLabel={t.timelineEmpty} layerLabel={t.layer} tokenLabel={t.timelineToken} />
          <div className="chart-pair">
            <TokenLine trace={timeline} metric="safety" title={t.safety} color={SERIES_SAFETY} emptyLabel={t.timelineEmpty} tokenLabel={t.timelineToken} format={(value) => `${Math.round(value * 100)}%`} />
            <TokenLine trace={timeline} metric="entropy" title={t.entropy} color={SERIES_ENTROPY} emptyLabel={t.timelineEmpty} tokenLabel={t.timelineToken} format={(value) => value.toFixed(2)} />
          </div>
        </section>

        <section className="card span-2">
          <PanelTitle icon={<BrainCircuit size={16} />} title={t.conceptMap} />
          <p className="card-hint">{t.conceptMapHint}</p>
          <ConceptMap trace={concepts} emptyLabel={t.conceptMapEmpty} layerLabel={t.layer} peakLabel={t.conceptPeak} conceptLabels={conceptLabels} />
        </section>

        <section className="card span-2">
          <PanelTitle icon={<BrainCircuit size={16} />} title={t.layerActivity} aside={layerOpCount ? <button className="ghost" onClick={onClearLayerOps} title={t.layerOpsClearTitle}><RotateCcw size={13} /> {layerOpCount}</button> : null} />
          <div className="layer-brush">
            <span className="brush-label">{t.layerOpsBrush}</span>
            <div className="brush-actions">
              {(["mute", "scale", "boost"] as const).map((action) => <button key={action} className={`chip${brushAction === action ? " on" : ""}`} onClick={() => onBrushActionChange(action)} title={t.layerOpsHelp[action]}>{action === "mute" ? t.mute : action === "scale" ? t.scaleAction : t.boost}</button>)}
            </div>
            {brushAction !== "mute" ? <label className="brush-scale"><span>{t.scale}</span><input type="range" min={0} max={3} step={0.05} value={brushScale} onChange={(event) => onBrushScaleChange(Number(event.target.value))} /><strong>×{brushScale.toFixed(2)}</strong></label> : null}
            <span className="brush-hint">{t.layerOpsHint}</span>
          </div>
          <LayerGrid layers={layers} layerCount={layerCount} activityLabel={t.activityTooltip} safetyLabel={t.safety} uncertaintyLabel={t.uncertainty || "Uncertainty"} ops={layerOps} onToggle={onToggleLayer} opHint={t.layerOpsCellHint} />
          {Object.keys(layerOps).some((layer) => Number(layer) <= 3) ? <p className="warn-note">{t.layerOpsEarlyWarning}</p> : null}
        </section>

        <section className="card span-2"><PanelTitle icon={<Eye size={16} />} title={t.layerLens} /><p className="card-hint">{t.layerLensHint}</p><LayerLensView items={lens} emptyLabel={t.noLens} /></section>
        <section className="card"><PanelTitle icon={<ShieldAlert size={16} />} title={t.safetyTrace} /><SafetyView safety={safety} language={language} /></section>
        <section className="card"><PanelTitle icon={<Activity size={16} />} title={t.topK} /><TopKList items={topK} emptyLabel={t.noCandidates} spaceLabel={t.spaceToken} /></section>
        <section className="card span-2"><PanelTitle icon={<Grid3x3 size={16} />} title={t.headMap} /><HeadMapView map={headMap} muted={mutedHeads} onToggle={onToggleHead} emptyLabel={t.noHeadMap} mutedLabel={t.headMapMuted} /></section>
        <section className="card"><PanelTitle icon={<Waves size={16} />} title={t.attention} /><AttentionView trace={attention} emptyLabel={t.noAttention} /></section>
        <section className="card"><PanelTitle icon={<Waves size={16} />} title={t.thinkPhase} /><ThinkPhaseView summary={thinkPhase} currentPhase={currentPhase} t={t} /></section>
        <section className="card span-2"><PanelTitle icon={<Activity size={16} />} title={t.runtime} /><RuntimeView metrics={blackBoxMetrics} log={log} language={language} /></section>
      </div>
    </div>
  );
}
