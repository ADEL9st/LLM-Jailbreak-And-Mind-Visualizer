import { describe, expect, it } from "vitest";

import { MAX_LAYER, buildInterventions, countRuleLayers, parseLayerSet, type LayerOp, type UIRule } from "./interventions";

const rule = (overrides: Partial<UIRule> = {}): UIRule => ({
  enabled: true,
  layerSet: "12",
  action: "mute",
  scale: 1,
  ...overrides
});

describe("parseLayerSet", () => {
  it("reads single layers", () => {
    expect(parseLayerSet("12")).toEqual([12]);
    expect(parseLayerSet("0")).toEqual([0]);
  });

  it("expands a range inclusively at both ends", () => {
    expect(parseLayerSet("10-13")).toEqual([10, 11, 12, 13]);
  });

  it("accepts commas, spaces, or both as separators", () => {
    expect(parseLayerSet("1,2 3")).toEqual([1, 2, 3]);
    expect(parseLayerSet("1 ,  2,3")).toEqual([1, 2, 3]);
  });

  it("mixes singles and ranges", () => {
    expect(parseLayerSet("8, 12, 16-18")).toEqual([8, 12, 16, 17, 18]);
  });

  it("reads a reversed range as the same span", () => {
    expect(parseLayerSet("18-16")).toEqual([16, 17, 18]);
  });

  it("de-duplicates overlapping input and returns sorted order", () => {
    expect(parseLayerSet("5, 3-6, 4")).toEqual([3, 4, 5, 6]);
  });

  it("returns an empty list for empty or whitespace input", () => {
    expect(parseLayerSet("")).toEqual([]);
    expect(parseLayerSet("   ")).toEqual([]);
    expect(parseLayerSet(",, ,")).toEqual([]);
  });

  it("drops garbage instead of throwing — it runs on every keystroke", () => {
    expect(parseLayerSet("abc")).toEqual([]);
    expect(parseLayerSet("1, abc, 3")).toEqual([1, 3]);
    expect(parseLayerSet("--")).toEqual([]);
    expect(parseLayerSet("1-")).toEqual([]);
  });

  it("rejects non-integers and negatives", () => {
    expect(parseLayerSet("1.5")).toEqual([]);
    expect(parseLayerSet("-3")).toEqual([]);
  });

  it("clamps at MAX_LAYER so a typo cannot generate thousands of rules", () => {
    expect(parseLayerSet("9999")).toEqual([]);
    expect(parseLayerSet(String(MAX_LAYER))).toEqual([MAX_LAYER]);
    expect(parseLayerSet(`250-9999`)).toEqual([250, 251, 252, 253, 254, 255]);
  });
});

describe("countRuleLayers", () => {
  it("counts the expanded layers, not the rules", () => {
    expect(countRuleLayers([rule({ layerSet: "10-13" })])).toBe(4);
  });

  it("ignores disabled and no-op rules", () => {
    expect(countRuleLayers([
      rule({ layerSet: "1-5", enabled: false }),
      rule({ layerSet: "1-5", action: "none" }),
      rule({ layerSet: "9" })
    ])).toBe(1);
  });

  it("is zero for no rules", () => {
    expect(countRuleLayers([])).toBe(0);
  });
});

describe("buildInterventions", () => {
  it("expands a range rule into one entry per layer", () => {
    const out = buildInterventions([rule({ layerSet: "10-12", action: "scale", scale: 0.5 })], {}, []);
    expect(out).toHaveLength(3);
    expect(out.map((item) => item.layer)).toEqual([10, 11, 12]);
    expect(out.every((item) => item.action === "scale" && item.scale === 0.5)).toBe(true);
    expect(out.every((item) => item.target_type === "layer" && item.head === null)).toBe(true);
  });

  it("skips disabled and no-op rules", () => {
    expect(buildInterventions([
      rule({ enabled: false }),
      rule({ action: "none" })
    ], {}, [])).toEqual([]);
  });

  it("includes click-to-intervene layer ops with their own action and strength", () => {
    const ops: Record<number, LayerOp> = { 14: { action: "boost", scale: 2 } };
    const out = buildInterventions([], ops, []);
    expect(out).toEqual([
      { enabled: true, target_type: "layer", layer: 14, head: null, action: "boost", scale: 2 }
    ]);
  });

  it("converts object keys back to numeric layer indices", () => {
    // Object.entries stringifies keys; shipping "14" instead of 14 would fail
    // the backend's schema.
    const out = buildInterventions([], { 14: { action: "mute", scale: 1 } }, []);
    expect(typeof out[0].layer).toBe("number");
  });

  it("turns muted head keys into head rules", () => {
    const out = buildInterventions([], {}, ["23:11", "24:3"]);
    expect(out).toEqual([
      { enabled: true, target_type: "head", layer: 23, head: 11, action: "mute", scale: 1 },
      { enabled: true, target_type: "head", layer: 24, head: 3, action: "mute", scale: 1 }
    ]);
  });

  it("drops a malformed head key rather than sending NaN", () => {
    expect(buildInterventions([], {}, ["bad", "1:", ":2", "5:6"])).toEqual([
      { enabled: true, target_type: "head", layer: 5, head: 6, action: "mute", scale: 1 }
    ]);
  });

  it("merges all three sources in a stable order", () => {
    const out = buildInterventions(
      [rule({ layerSet: "1", action: "mute" })],
      { 14: { action: "boost", scale: 2 } },
      ["23:11"]
    );
    expect(out.map((item) => `${item.target_type}:${item.layer}`)).toEqual(["layer:1", "layer:14", "head:23"]);
  });

  it("keeps an overlap between a range rule and a clicked layer as two entries", () => {
    // The backend's layer_factor compounds duplicates (mute wins, scales
    // multiply); dropping one here would silently change the result.
    const out = buildInterventions(
      [rule({ layerSet: "14", action: "scale", scale: 0.5 })],
      { 14: { action: "scale", scale: 0.5 } },
      []
    );
    expect(out).toHaveLength(2);
    expect(out.every((item) => item.layer === 14)).toBe(true);
  });

  it("returns an empty list when nothing is configured", () => {
    expect(buildInterventions([], {}, [])).toEqual([]);
  });

  it("accepts a Set for muted heads, matching the component state", () => {
    expect(buildInterventions([], {}, new Set(["7:2"]))).toHaveLength(1);
  });
});
