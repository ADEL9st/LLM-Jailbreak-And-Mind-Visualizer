/** The steering modes, ordered by how invasive they are.
 *
 *  A flat dropdown of 14 names hides the fact that they vary along only two
 *  axes — *mechanism* and *strength* — and gives no clue where to start. This
 *  arranges them as a ladder: begin at the top, move down only if a run does
 *  not flip.
 *
 *  The `measured` numbers come from a sweep on qwen2.5-1.5b, 40 tokens, temp 0,
 *  one hard-refusal prompt. `peak` = refusal signal still present after
 *  steering (lower is a cleaner ablation); `coherence` = fraction of the output
 *  that reads as ordinary words. One model and one prompt, so treat them as a
 *  starting hint rather than a benchmark — but they are measured, not assumed.
 *
 *  Headline finding: 13 of 14 flipped the refusal, and the *softest* mode left
 *  the least refusal signal behind. Escalating mostly costs coherence without
 *  buying extra bypass.
 */

import type { JailbreakMode } from "./types";

export type ModeTier = "soft" | "medium" | "strong" | "experimental" | "destructive";

export interface ModeMeasurement {
  peak: number;
  coherence: number;
  flipped: boolean;
}

export interface ModeInfo {
  mode: JailbreakMode;
  tier: ModeTier;
  /** False when the mode is the same maths as another, only at a different
   *  constant — worth knowing before you try it expecting something new. */
  distinctMechanism: boolean;
  /** i18n key describing what it actually does. */
  summaryKey: string;
  measured?: ModeMeasurement;
}

/** Ordered least → most invasive. The UI renders them in this order. */
export const MODE_LADDER: ModeInfo[] = [
  { mode: "default", tier: "soft", distinctMechanism: true, summaryKey: "modeSoftDefault",
    measured: { peak: 0.21, coherence: 1.0, flipped: true } },
  { mode: "pid_control", tier: "soft", distinctMechanism: true, summaryKey: "modePid",
    measured: { peak: 0.20, coherence: 1.0, flipped: true } },
  { mode: "orthogonal_steer", tier: "soft", distinctMechanism: true, summaryKey: "modeOrthogonal",
    measured: { peak: 0.26, coherence: 1.0, flipped: true } },

  { mode: "advanced", tier: "medium", distinctMechanism: false, summaryKey: "modeAdvanced",
    measured: { peak: 0.36, coherence: 1.0, flipped: true } },
  { mode: "caa_dynamic", tier: "medium", distinctMechanism: true, summaryKey: "modeCaa",
    measured: { peak: 0.36, coherence: 0.97, flipped: true } },
  { mode: "adaptive_steer", tier: "medium", distinctMechanism: true, summaryKey: "modePid" },

  { mode: "broker_math", tier: "strong", distinctMechanism: false, summaryKey: "modeBrokerMath",
    measured: { peak: 0.35, coherence: 1.0, flipped: true } },
  { mode: "broker_half", tier: "strong", distinctMechanism: false, summaryKey: "modeBrokerHalf",
    measured: { peak: 0.33, coherence: 1.0, flipped: true } },
  { mode: "broker_full", tier: "strong", distinctMechanism: false, summaryKey: "modeBrokerFull",
    measured: { peak: 0.35, coherence: 1.0, flipped: true } },
  { mode: "gradient_steer", tier: "strong", distinctMechanism: false, summaryKey: "modeGradient",
    measured: { peak: 0.32, coherence: 0.97, flipped: true } },
  { mode: "progressive", tier: "strong", distinctMechanism: false, summaryKey: "modeProgressive",
    measured: { peak: 0.32, coherence: 0.93, flipped: true } },

  { mode: "activation_patch", tier: "experimental", distinctMechanism: true, summaryKey: "modeActivationPatch",
    measured: { peak: 0.97, coherence: 1.0, flipped: true } },
  { mode: "commit_release", tier: "experimental", distinctMechanism: true, summaryKey: "modeTokenWindow" },
  { mode: "token_window", tier: "experimental", distinctMechanism: true, summaryKey: "modeTokenWindow",
    measured: { peak: 0.54, coherence: 1.0, flipped: true } },
  { mode: "mlp_clamp", tier: "experimental", distinctMechanism: true, summaryKey: "modeMlpClamp",
    measured: { peak: 0.45, coherence: 0.97, flipped: true } },

  { mode: "surgical", tier: "destructive", distinctMechanism: true, summaryKey: "modeSurgical",
    measured: { peak: 0.0, coherence: 0.0, flipped: false } }
];

export const TIER_ORDER: ModeTier[] = ["soft", "medium", "strong", "experimental", "destructive"];

export function infoFor(mode: JailbreakMode): ModeInfo | undefined {
  return MODE_LADDER.find((item) => item.mode === mode);
}

export function modesInTier(tier: ModeTier): ModeInfo[] {
  return MODE_LADDER.filter((item) => item.tier === tier);
}

/** The mode to start from — softest, and measured to work at least as well as
 *  anything harsher. */
export const RECOMMENDED_MODE: JailbreakMode = "default";

/** True when a mode is redundant with something gentler: same mechanism, only a
 *  larger constant, and no better measured result. */
export function isRedundant(info: ModeInfo): boolean {
  if (info.distinctMechanism || !info.measured) return false;
  const softestPeak = Math.min(...modesInTier("soft").flatMap((item) => item.measured ? [item.measured.peak] : []));
  return info.measured.peak >= softestPeak;
}
