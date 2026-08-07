/** What each adapter can actually deliver.
 *
 *  Without this the UI lies by omission: pick `transformers` or an API adapter
 *  and the Analysis page renders a wall of empty cards with no explanation. The
 *  panels use this to say *why* they are empty and what to switch to.
 *
 *  Derived from what each adapter emits (checked against the backend, not
 *  guessed) — keep it in sync when an adapter gains or loses an event.
 */

import type { AdapterName } from "./types";

/** Telemetry features a panel can depend on. */
export type Capability =
  | "layerActivity"
  | "safetyTrace"
  | "logitLens"
  | "headMap"
  | "concepts"
  | "thinkPhase"
  | "timeline";

export const ALL_CAPABILITIES: Capability[] = [
  "layerActivity",
  "safetyTrace",
  "logitLens",
  "headMap",
  "concepts",
  "thinkPhase",
  "timeline"
];

export interface AdapterProfile {
  /** full = every panel works · partial = some · none = no internals at all. */
  tier: "full" | "partial" | "none";
  capabilities: Capability[];
  /** i18n key explaining the gap, shown when tier !== "full". */
  reasonKey: "adapterCapLegacy" | "adapterCapBlackBox" | "adapterCapApi" | "adapterCapMock" | "adapterCapNoConcepts" | null;
}

const FULL: Capability[] = ALL_CAPABILITIES;

export const ADAPTER_PROFILES: Record<AdapterName, AdapterProfile> = {
  pytorch: { tier: "full", capabilities: FULL, reasonKey: null },
  // nnsight emits everything pytorch does EXCEPT the concept map — that was only
  // wired into the pytorch adapter. Claiming "full" here made the Analysis page
  // show an unexplained empty Concept Map card on nnsight runs.
  nnsight: {
    tier: "partial",
    capabilities: ALL_CAPABILITIES.filter((c) => c !== "concepts"),
    reasonKey: "adapterCapNoConcepts"
  },
  // Legacy hook engine: only ever emitted layer_activity.
  transformers: { tier: "partial", capabilities: ["layerActivity"], reasonKey: "adapterCapLegacy" },
  // Scripted telemetry — every panel is populated, but none of it is real.
  mock: { tier: "partial", capabilities: FULL, reasonKey: "adapterCapMock" },
  // Separate runtime, no hooks reachable.
  ollama: { tier: "none", capabilities: [], reasonKey: "adapterCapBlackBox" },
  openai: { tier: "none", capabilities: [], reasonKey: "adapterCapApi" },
  anthropic: { tier: "none", capabilities: [], reasonKey: "adapterCapApi" },
  gemini: { tier: "none", capabilities: [], reasonKey: "adapterCapApi" }
};

export function profileFor(adapter: AdapterName): AdapterProfile {
  return ADAPTER_PROFILES[adapter] ?? ADAPTER_PROFILES.mock;
}

export function supports(adapter: AdapterName, capability: Capability): boolean {
  return profileFor(adapter).capabilities.includes(capability);
}

/** Capabilities this adapter cannot provide — what to tell the user is missing. */
export function missingCapabilities(adapter: AdapterName): Capability[] {
  const have = new Set(profileFor(adapter).capabilities);
  return ALL_CAPABILITIES.filter((capability) => !have.has(capability));
}
