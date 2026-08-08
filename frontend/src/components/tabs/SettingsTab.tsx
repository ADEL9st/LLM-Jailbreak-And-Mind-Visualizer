import { Plus, RotateCcw, Trash2 } from "lucide-react";
import { compactNumber, jailbreakModeLabel, parseNumberList, researchPresetCopy } from "../../app/benchmarking";
import { RECOMMENDED_MODE, TIER_ORDER, isRedundant, modesInTier } from "../../jailbreakModes";
import { RESEARCH_PRESETS } from "../../presets";
import { languageOptions, type Language, type Translation } from "../../i18n";
import type { UIRule } from "../../interventions";
import type {
  AdapterName,
  InterventionAction,
  JailbreakMode,
  ModelInfo,
  OutputPolicy,
  PromptCraftType,
  Quantization,
  SteeringOptions,
  TokenLimitMode
} from "../../types";

type SteeringTargetMode = "automatic" | "window" | "layers" | "depths";

interface SettingsTabProps {
  t: Translation;
  adapter: AdapterName;
  onAdapterChange: (adapter: AdapterName) => void;
  model: string;
  onModelChange: (model: string) => void;
  modelOptions: ModelInfo[];
  selectedModel?: ModelInfo;
  isApiAdapter: boolean;
  apiKey: string;
  onApiKeyChange: (value: string) => void;
  whiteboxAdapter: boolean;
  onUnloadModels: () => void;
  running: boolean;
  tokenLimitMode: TokenLimitMode;
  onTokenLimitModeChange: (mode: TokenLimitMode) => void;
  maxTokens: number;
  onMaxTokensChange: (value: number) => void;
  temperature: number;
  onTemperatureChange: (value: number) => void;
  outputPolicy: OutputPolicy;
  quantization: Quantization;
  onQuantizationChange: (value: Quantization) => void;
  effectiveMaxTokens: number | null;
  contextLength: number | null;
  hardwareSafeMaxTokens: number | null;
  systemPrompt: string;
  onSystemPromptChange: (value: string) => void;
  assistantPrefill: string;
  onAssistantPrefillChange: (value: string) => void;
  promptCraft: PromptCraftType;
  onPromptCraftChange: (value: PromptCraftType) => void;
  selectedPreset: string;
  onApplyPreset: (id: string) => void;
  jailbreak: boolean;
  onJailbreakChange: (value: boolean) => void;
  jailbreakMode: JailbreakMode;
  onJailbreakModeChange: (mode: JailbreakMode) => void;
  useMlpAblation: boolean;
  onUseMlpAblationChange: (value: boolean) => void;
  useHelpfulnessBoost: boolean;
  onUseHelpfulnessBoostChange: (value: boolean) => void;
  useNormRegulation: boolean;
  onUseNormRegulationChange: (value: boolean) => void;
  useDiversionSuppression: boolean;
  onUseDiversionSuppressionChange: (value: boolean) => void;
  steering: SteeringOptions;
  steeringTargetMode: SteeringTargetMode;
  onSteeringTargetModeChange: (mode: SteeringTargetMode) => void;
  onUpdateSteering: (patch: Partial<SteeringOptions>) => void;
  onResetSteering: () => void;
  activeInterventionCount: number;
  interventions: UIRule[];
  onAddIntervention: () => void;
  onUpdateIntervention: (index: number, patch: Partial<UIRule>) => void;
  onRemoveIntervention: (index: number) => void;
  mutedHeadCount: number;
  onClearMutedHeads: () => void;
  language: Language;
  onLanguageChange: (language: Language) => void;
}

export function SettingsTab(props: SettingsTabProps) {
  const {
    t, adapter, onAdapterChange, model, onModelChange, modelOptions, selectedModel,
    isApiAdapter, apiKey, onApiKeyChange, whiteboxAdapter, onUnloadModels, running,
    tokenLimitMode, onTokenLimitModeChange, maxTokens, onMaxTokensChange, temperature,
    onTemperatureChange, outputPolicy, quantization, onQuantizationChange,
    effectiveMaxTokens, contextLength, hardwareSafeMaxTokens, systemPrompt,
    onSystemPromptChange, assistantPrefill, onAssistantPrefillChange, promptCraft,
    onPromptCraftChange, selectedPreset, onApplyPreset, jailbreak, onJailbreakChange,
    jailbreakMode, onJailbreakModeChange, useMlpAblation, onUseMlpAblationChange,
    useHelpfulnessBoost, onUseHelpfulnessBoostChange, useNormRegulation,
    onUseNormRegulationChange, useDiversionSuppression, onUseDiversionSuppressionChange,
    steering, steeringTargetMode, onSteeringTargetModeChange, onUpdateSteering,
    onResetSteering, activeInterventionCount, interventions, onAddIntervention,
    onUpdateIntervention, onRemoveIntervention, mutedHeadCount, onClearMutedHeads,
    language, onLanguageChange
  } = props;

  return (
    <div className="settings-page">
      <section className="settings-group">
        <h2>{t.settingsEngine}</h2>
        <p className="group-hint">{t.settingsEngineHint}</p>
        <div className="field-grid">
          <label className="field">
            <span>{t.adapter}</span>
            <select value={adapter} onChange={(event) => onAdapterChange(event.target.value as AdapterName)}>
              <option value="mock">{t.adapterMock}</option><option value="ollama">{t.adapterOllama}</option>
              <option value="nnsight">{t.adapterNnsight}</option><option value="pytorch">{t.adapterPytorch}</option>
              <option value="transformers">{t.adapterTransformers}</option><option value="openai">{t.adapterOpenai}</option>
              <option value="anthropic">{t.adapterAnthropic}</option><option value="gemini">{t.adapterGemini}</option>
            </select>
          </label>
          <label className="field">
            <span>{t.model}</span>
            <select value={model} onChange={(event) => onModelChange(event.target.value)}>
              {modelOptions.map((item) => <option key={`${item.adapter}-${item.id}`} value={item.id}>{item.label}</option>)}
            </select>
          </label>
        </div>
        {selectedModel ? (
          <div className="model-profile">
            <p className="group-hint">{selectedModel.description}</p>
            <div className="metric-row wrap">
              <span>{t.ui.modelType} <strong>{selectedModel.model_type || "—"}</strong></span>
              <span>{t.ui.modelLayers} <strong>{selectedModel.layer_count ?? "—"}</strong></span>
              <span>{t.ui.modelHidden} <strong>{compactNumber(selectedModel.hidden_size)}</strong></span>
              <span>{t.ui.modelHeads} <strong>{selectedModel.attention_heads ?? "—"}</strong></span>
              <span>{t.ui.modelContext} <strong>{compactNumber(selectedModel.context_length)}</strong></span>
              <span>{t.ui.modelDtype} <strong>{selectedModel.dtype || "—"}</strong></span>
              <span>{t.ui.autoProbe} <strong>{selectedModel.compatibility?.status || "runtime"}</strong></span>
            </div>
            {selectedModel.compatibility?.warnings?.length ? <p className="group-hint">{selectedModel.compatibility.warnings.join(" ")}</p> : null}
          </div>
        ) : null}
        {isApiAdapter ? <><label className="field"><span>{t.apiKeyLabel}</span><input type="password" placeholder={adapter === "anthropic" ? "sk-ant-..." : adapter === "gemini" ? "AIza..." : "sk-..."} value={apiKey} onChange={(event) => onApiKeyChange(event.target.value)} /></label><p className="warn-note">{t.apiAdapterWarning}</p></> : null}
        {whiteboxAdapter ? <div className="group-actions"><button className="ghost" onClick={onUnloadModels} disabled={running} title={t.unloadModelTitle}>{t.unloadModel}</button></div> : null}
      </section>

      <section className="settings-group">
        <h2>{t.settingsGeneration}</h2>
        <div className="field-grid">
          <label className="field"><span>{t.ui.tokenBudgetMode}</span><select value={tokenLimitMode} onChange={(event) => onTokenLimitModeChange(event.target.value as TokenLimitMode)}><option value="fixed">{t.ui.fixedLimit}</option><option value="model">{t.ui.modelWindowAutomatic}</option></select></label>
          <label className="field"><span>{t.maxTokens}</span><input type="number" min={1} max={65536} disabled={tokenLimitMode === "model"} value={maxTokens} onChange={(event) => onMaxTokensChange(Math.max(1, Math.min(65536, Number(event.target.value))))} /></label>
          <label className="field"><span>{t.temperature}</span><input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(event) => onTemperatureChange(Number(event.target.value))} /></label>
          <label className="field"><span>{t.output}</span><select value={outputPolicy} disabled><option value="raw">{t.outputRaw}</option></select></label>
          {whiteboxAdapter ? <label className="field"><span>{t.precision}</span><select value={quantization} onChange={(event) => onQuantizationChange(event.target.value as Quantization)}><option value="none">{t.precisionFull}</option><option value="4bit">{t.precision4bit}</option><option value="8bit">{t.precision8bit}</option></select></label> : null}
        </div>
        <p className="group-hint">
          {tokenLimitMode === "model" ? `${t.ui.autoBudgetHint}${selectedModel?.context_length ? ` (${t.ui.modelWindow}: ${selectedModel.context_length.toLocaleString()} ${t.ui.tokens})` : ""} ${t.ui.notInfinite}` : t.ui.fixedBudgetHint}
          {effectiveMaxTokens !== null ? ` ${t.ui.lastRunBudget}: ${effectiveMaxTokens.toLocaleString()} / ${t.ui.context} ${contextLength?.toLocaleString() ?? t.ui.unknown}.` : ""}
          {hardwareSafeMaxTokens !== null ? ` ${t.ui.vramSafeEstimate}: ${hardwareSafeMaxTokens.toLocaleString()}.` : ""}
        </p>
        <p className="group-hint">{t.ui.rawOutputHint}</p>
        {whiteboxAdapter ? <p className="group-hint">{t.precisionHint}</p> : null}
      </section>

      <section className="settings-group">
        <h2>{t.settingsPromptLab}</h2>
        <label className="field"><span>{t.ui.systemPrompt}</span><textarea rows={4} value={systemPrompt} onChange={(event) => onSystemPromptChange(event.target.value)} placeholder={t.ui.systemPromptPlaceholder} /></label>
        <label className="field"><span>{t.ui.assistantPrefill}</span><textarea rows={3} value={assistantPrefill} onChange={(event) => onAssistantPrefillChange(event.target.value)} placeholder={t.ui.assistantPrefillPlaceholder} /></label>
        <label className="field"><span>{t.promptCraftLabel}</span><select value={promptCraft} onChange={(event) => onPromptCraftChange(event.target.value as PromptCraftType)}><option value="none">{t.promptCraftNone}</option><option value="base64">{t.promptCraftBase64}</option><option value="rot13">{t.promptCraftRot13}</option><option value="leetspeak">{t.promptCraftLeetspeak}</option><option value="dan">{t.promptCraftDan}</option><option value="developer">{t.promptCraftDeveloper}</option><option value="crescendo">{t.promptCraftCrescendo}</option><option value="aim">{t.promptCraftAim}</option><option value="indirect_injection">{t.promptCraftIndirectInjection}</option><option value="many_shot">{t.promptCraftManyShot}</option><option value="gcg_suffix">{t.promptCraftGcgSuffix}</option><option value="virtualization">{t.promptCraftVirtualization}</option></select></label>
        {promptCraft !== "none" ? <p className="group-hint">{t.ui.promptCraftHints[promptCraft]}</p> : null}
      </section>

      <section className="settings-group">
        <h2>{t.settingsJailbreak}</h2>
        <label className="field"><span>{t.ui.researchPreset}</span><select value={selectedPreset} onChange={(event) => onApplyPreset(event.target.value)}><option value="custom">{t.ui.custom}</option>{RESEARCH_PRESETS.map((preset) => <option key={preset.id} value={preset.id}>{researchPresetCopy(preset.id, t).label}</option>)}</select></label>
        {selectedPreset !== "custom" ? <p className="group-hint">{researchPresetCopy(selectedPreset, t).description}</p> : null}
        <label className="toggle-row" title={t.jailbreakHint}><input type="checkbox" checked={jailbreak} onChange={(event) => onJailbreakChange(event.target.checked)} /><span className="toggle-track"><i /></span><span className="toggle-text">{t.jailbreak}</span></label>
        <p className="group-hint">{t.jailbreakHint}</p>
        {jailbreak ? <><label className="field"><span>{t.jbLadderTitle}</span></label><p className="group-hint">{t.jbLadderHint}</p><div className="mode-ladder">
          {TIER_ORDER.map((tier) => {
            const modes = modesInTier(tier);
            if (!modes.length) return null;
            return <div className={`mode-tier tier-${tier}`} key={tier}><span className="mode-tier-label">{t[`jbTier${tier[0].toUpperCase()}${tier.slice(1)}` as keyof typeof t] as string}</span>{modes.map((info) => <button key={info.mode} type="button" className={`mode-row${jailbreakMode === info.mode ? " selected" : ""}`} onClick={() => onJailbreakModeChange(info.mode)}><span className="mode-row-head"><strong>{jailbreakModeLabel(info.mode, t)}</strong>{info.mode === RECOMMENDED_MODE ? <em className="mode-badge good">{t.jbRecommended}</em> : null}{isRedundant(info) ? <em className="mode-badge dim">{t.jbRedundant}</em> : null}</span><span className="mode-row-body">{t[info.summaryKey as keyof typeof t] as string}</span>{info.measured ? <span className="mode-row-measured">{t.jbMeasuredPeak} {Math.round(info.measured.peak * 100)}% · {t.jbMeasuredCoh} {Math.round(info.measured.coherence * 100)}%</span> : <span className="mode-row-measured">{t.ui.notBenchmarked}</span>}</button>)}</div>;
          })}
          <p className="group-hint mode-measured-note">{t.jbMeasuredNote}</p>
        </div><div className="check-stack"><label className="check-row" title={t.steerMlpHint}><input type="checkbox" checked={useMlpAblation} onChange={(event) => onUseMlpAblationChange(event.target.checked)} /><span>{t.steerMlpLabel}</span></label><label className="check-row" title={t.steerHelpHint}><input type="checkbox" checked={useHelpfulnessBoost} onChange={(event) => onUseHelpfulnessBoostChange(event.target.checked)} /><span>{t.steerHelpLabel}</span></label><label className="check-row" title={t.steerNormHint}><input type="checkbox" checked={useNormRegulation} onChange={(event) => onUseNormRegulationChange(event.target.checked)} /><span>{t.steerNormLabel}</span></label><label className="check-row" title={t.steerDiversionHint}><input type="checkbox" checked={useDiversionSuppression} onChange={(event) => onUseDiversionSuppressionChange(event.target.checked)} /><span>{t.steerDiversionLabel}</span></label></div></> : null}
      </section>

      <section className="settings-group">
        <div className="group-head"><div><h2>{t.ui.advancedSteering}</h2><p className="group-hint">{t.ui.advancedSteeringHint}</p></div><button className="ghost" type="button" onClick={onResetSteering}>{t.ui.reset}</button></div>
        <div className="field-grid three"><label className="field"><span>{t.ui.strength}</span><input type="number" min={0} max={5} step={0.05} value={steering.strength} onChange={(event) => onUpdateSteering({ strength: Number(event.target.value) })} /></label><label className="field"><span>{t.ui.maximumLayers}</span><input type="number" min={1} max={128} disabled={steering.all_layers} value={steering.max_layers} onChange={(event) => onUpdateSteering({ max_layers: Number(event.target.value) })} /></label><label className="field"><span>{t.ui.layerTargeting}</span><select value={steeringTargetMode} onChange={(event) => onSteeringTargetModeChange(event.target.value as SteeringTargetMode)}><option value="automatic">{t.ui.automaticCalibration}</option><option value="window">{t.ui.relativeDepthWindow}</option><option value="layers">{t.ui.exactLayers}</option><option value="depths">{t.ui.exactRelativeDepths}</option></select></label></div>
        <div className="check-stack"><label className="check-row"><input type="checkbox" checked={steering.all_layers} onChange={(event) => onUpdateSteering({ all_layers: event.target.checked })} /><span>{t.ui.allowAllTargetedLayers}</span></label><label className="check-row"><input type="checkbox" checked={steering.primary_only} onChange={(event) => onUpdateSteering({ primary_only: event.target.checked })} /><span>{t.ui.primaryRefusalOnly}</span></label><label className="check-row"><input type="checkbox" checked={steering.coherence_recovery} onChange={(event) => onUpdateSteering({ coherence_recovery: event.target.checked })} /><span>{t.ui.coherenceRecovery}</span></label></div>
        {steering.use_depth_window ? <div className="field-grid"><label className="field"><span>{t.ui.depthStart}</span><input type="number" min={0} max={0.99} step={0.05} value={steering.depth_start} onChange={(event) => onUpdateSteering({ depth_start: Number(event.target.value) })} /></label><label className="field"><span>{t.ui.depthEnd}</span><input type="number" min={0.01} max={1} step={0.05} value={steering.depth_end} onChange={(event) => onUpdateSteering({ depth_end: Number(event.target.value) })} /></label></div> : null}
        {steeringTargetMode === "layers" ? <label className="field"><span>{t.ui.exactLayersCsv}</span><input value={steering.target_layers.join(", ")} onChange={(event) => onUpdateSteering({ target_layers: parseNumberList(event.target.value).filter((value) => Number.isInteger(value) && value >= 0 && value <= 1023), target_depths: [], use_depth_window: false })} placeholder="18, 22, 26" /></label> : null}
        {steeringTargetMode === "depths" ? <label className="field"><span>{t.ui.relativeDepthsCsv}</span><input value={steering.target_depths.join(", ")} onChange={(event) => onUpdateSteering({ target_depths: parseNumberList(event.target.value).filter((value) => value >= 0 && value <= 1), target_layers: [], use_depth_window: false })} placeholder="0.65, 0.8, 0.95" /></label> : null}
        <details><summary>{t.ui.modeMultipliers}</summary><div className="field-grid three"><label className="field"><span>{t.ui.diversionPenalty}</span><input type="number" min={0} max={50} step={0.5} value={steering.diversion_penalty} onChange={(event) => onUpdateSteering({ diversion_penalty: Number(event.target.value) })} /></label><label className="field"><span>{t.ui.patchThroughStep}</span><input type="number" min={0} max={2048} value={steering.patch_last_step} onChange={(event) => onUpdateSteering({ patch_last_step: Number(event.target.value) })} /></label><label className="field"><span>{t.ui.patchMultiplier}</span><input type="number" min={0} max={10} step={0.1} value={steering.patch_multiplier} onChange={(event) => onUpdateSteering({ patch_multiplier: Number(event.target.value) })} /></label><label className="field"><span>{t.ui.commitSteps}</span><input type="number" min={0} max={2048} value={steering.commit_steps} onChange={(event) => onUpdateSteering({ commit_steps: Number(event.target.value) })} /></label><label className="field"><span>{t.ui.commitMultiplier}</span><input type="number" min={0} max={10} step={0.1} value={steering.commit_multiplier} onChange={(event) => onUpdateSteering({ commit_multiplier: Number(event.target.value) })} /></label><label className="field"><span>{t.ui.maintenanceMultiplier}</span><input type="number" min={0} max={10} step={0.1} value={steering.maintenance_multiplier} onChange={(event) => onUpdateSteering({ maintenance_multiplier: Number(event.target.value) })} /></label></div></details>
      </section>

      <section className="settings-group">
        <div className="group-head"><div><h2>{t.interventionStack}</h2><p className="group-hint">{activeInterventionCount} {t.activeRules}</p></div><button className="ghost" type="button" onClick={onAddIntervention}><Plus size={14} /> {t.addRule}</button></div>
        {interventions.length ? <div className="rule-stack">{interventions.map((item, index) => <article className="rule" key={`rule-${index}`}><div className="rule-head"><label className="check-row"><input type="checkbox" checked={item.enabled} onChange={(event) => onUpdateIntervention(index, { enabled: event.target.checked })} /><span>{t.rule} {index + 1}</span></label><button className="ghost icon" type="button" onClick={() => onRemoveIntervention(index)} title={t.removeRule}><Trash2 size={14} /></button></div><div className="field-grid three"><label className="field"><span>{t.layerSet}</span><input type="text" value={item.layerSet} onChange={(event) => onUpdateIntervention(index, { layerSet: event.target.value })} placeholder="10-25, 28" /></label><label className="field"><span>{t.action}</span><select value={item.action} onChange={(event) => onUpdateIntervention(index, { action: event.target.value as InterventionAction })}><option value="none">{t.none}</option><option value="mute">{t.mute}</option><option value="scale">{t.scaleAction}</option><option value="boost">{t.boost}</option></select></label><label className="field"><span>{t.scale}</span><input type="number" min={0} max={3} step={0.05} value={item.scale} onChange={(event) => onUpdateIntervention(index, { scale: Number(event.target.value) })} /></label></div></article>)}</div> : <p className="group-hint">{t.noInterventions}</p>}
        {mutedHeadCount ? <div className="group-actions"><span className="group-hint">{mutedHeadCount} {t.headMapMuted}</span><button className="ghost" onClick={onClearMutedHeads}><RotateCcw size={14} /> {t.benchmarkClear}</button></div> : null}
      </section>

      <section className="settings-group"><h2>{t.language}</h2><div className="lang-row">{languageOptions.map((item) => <button key={item.code} className={`chip${language === item.code ? " on" : ""}`} onClick={() => onLanguageChange(item.code)}>{item.label}</button>)}</div></section>
    </div>
  );
}
