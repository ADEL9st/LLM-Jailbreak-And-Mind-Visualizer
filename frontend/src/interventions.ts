/** Turning the three ways the UI lets you express an intervention into the one
 *  flat list the backend accepts.
 *
 *  The three sources, all of which can be active at once:
 *    1. the Settings rule stack  — range expressions like "10-25, 28"
 *    2. layer ops                — click-to-intervene on the Layer Activity grid
 *    3. muted heads              — click-to-mute on the Head Map
 *
 *  Extracted from App.tsx so the parsing and the merge can be tested; a mistake
 *  here silently sends the wrong layers to the model, which is the hardest kind
 *  of bug to notice in this tool.
 */

import type { InterventionAction, InterventionConfig } from "./types";

/** A rule as the Settings panel holds it, before the layer set is expanded. */
export interface UIRule {
  enabled: boolean;
  layerSet: string;
  action: InterventionAction;
  scale: number;
}

export interface LayerOp {
  action: "mute" | "scale" | "boost";
  scale: number;
}

/** Highest layer index we will accept from free text — deep models top out well
 *  below this, and it keeps a typo like "9999" from generating 10k rules. */
export const MAX_LAYER = 255;

/**
 * Expand a layer-set expression into sorted, unique layer indices.
 *
 * Accepts comma- and/or whitespace-separated singles and ranges: "3", "10-25",
 * "8, 12 16-18". Reversed ranges ("25-10") are read as the same span. Anything
 * unparseable is dropped rather than throwing — this runs on every keystroke in
 * a text field.
 */
export function parseLayerSet(value: string): number[] {
  const layers = new Set<number>();
  const chunks = value
    .split(/[,\s]+/)
    .map((item) => item.trim())
    .filter(Boolean);

  for (const chunk of chunks) {
    const range = chunk.match(/^(\d+)-(\d+)$/);
    if (range) {
      const low = Math.min(Number(range[1]), Number(range[2]));
      const high = Math.max(Number(range[1]), Number(range[2]));
      for (let layer = low; layer <= high; layer += 1) {
        if (layer >= 0 && layer <= MAX_LAYER) layers.add(layer);
      }
      continue;
    }

    const layer = Number(chunk);
    if (Number.isInteger(layer) && layer >= 0 && layer <= MAX_LAYER) {
      layers.add(layer);
    }
  }

  return [...layers].sort((a, b) => a - b);
}

/** How many individual layers the rule stack currently targets. */
export function countRuleLayers(rules: UIRule[]): number {
  let count = 0;
  for (const rule of rules) {
    if (rule.enabled && rule.action !== "none") count += parseLayerSet(rule.layerSet).length;
  }
  return count;
}

function layerRule(layer: number, action: InterventionAction, scale: number): InterventionConfig {
  return { enabled: true, target_type: "layer", layer, head: null, action, scale };
}

/**
 * Merge every source into the request payload.
 *
 * Order matters: the rule stack goes first, then direct layer ops, then head
 * mutes. The backend's `layer_factor` folds duplicate layer rules together
 * (mute wins, scales multiply), so an overlap between a range rule and a
 * clicked layer compounds rather than one silently replacing the other.
 */
export function buildInterventions(
  rules: UIRule[],
  layerOps: Record<number | string, LayerOp>,
  mutedHeads: Iterable<string>
): InterventionConfig[] {
  const flat: InterventionConfig[] = [];

  for (const rule of rules) {
    if (!rule.enabled || rule.action === "none") continue;
    for (const layer of parseLayerSet(rule.layerSet)) {
      flat.push(layerRule(layer, rule.action, rule.scale));
    }
  }

  for (const [layer, op] of Object.entries(layerOps)) {
    flat.push(layerRule(Number(layer), op.action, op.scale));
  }

  for (const key of mutedHeads) {
    // Match the whole key rather than split+Number: Number("") is 0, not NaN, so
    // a truncated key like "1:" would otherwise become a real rule for head 0.
    const match = /^(\d+):(\d+)$/.exec(key);
    if (!match) continue;
    flat.push({
      enabled: true,
      target_type: "head",
      layer: Number(match[1]),
      head: Number(match[2]),
      action: "mute",
      scale: 1
    });
  }

  return flat;
}
