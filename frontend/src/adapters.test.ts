import { describe, expect, it } from "vitest";

import { ADAPTER_PROFILES, ALL_CAPABILITIES, missingCapabilities, profileFor, supports } from "./adapters";
import type { AdapterName } from "./types";

const ADAPTERS = Object.keys(ADAPTER_PROFILES) as AdapterName[];

describe("adapter profiles", () => {
  it("covers every adapter the UI offers", () => {
    // Drift here is how a new adapter ends up silently showing empty panels.
    expect(ADAPTERS.sort()).toEqual(
      ["anthropic", "gemini", "mock", "nnsight", "ollama", "openai", "pytorch", "transformers"].sort()
    );
  });

  it("pytorch is the only adapter that delivers every panel", () => {
    const full = ADAPTERS.filter((name) => ADAPTER_PROFILES[name].tier === "full");
    expect(full).toEqual(["pytorch"]);
  });

  it("nnsight is full except the concept map, which was never wired into it", () => {
    // Regression: this used to claim "full", so an nnsight run showed an empty
    // Concept Map card with no explanation.
    expect(supports("nnsight", "safetyTrace")).toBe(true);
    expect(supports("nnsight", "timeline")).toBe(true);
    expect(supports("nnsight", "concepts")).toBe(false);
    expect(missingCapabilities("nnsight")).toEqual(["concepts"]);
  });

  it("full-tier adapters need no explanation, everything else must have one", () => {
    for (const name of ADAPTERS) {
      const profile = ADAPTER_PROFILES[name];
      if (profile.tier === "full") expect(profile.reasonKey).toBeNull();
      else expect(profile.reasonKey).toBeTruthy();
    }
  });

  it("a tier-none adapter claims no capabilities", () => {
    for (const name of ADAPTERS) {
      if (ADAPTER_PROFILES[name].tier === "none") {
        expect(ADAPTER_PROFILES[name].capabilities).toEqual([]);
      }
    }
  });

  it("mock is flagged even though it fills every panel", () => {
    // It is the one adapter whose data looks complete but is invented, so the
    // notice matters more here than anywhere else.
    expect(ADAPTER_PROFILES.mock.capabilities).toEqual(ALL_CAPABILITIES);
    expect(ADAPTER_PROFILES.mock.reasonKey).toBe("adapterCapMock");
  });

  it("no profile claims a capability outside the known set", () => {
    for (const name of ADAPTERS) {
      for (const capability of ADAPTER_PROFILES[name].capabilities) {
        expect(ALL_CAPABILITIES).toContain(capability);
      }
    }
  });
});

describe("supports / missingCapabilities", () => {
  it("pytorch supports everything", () => {
    for (const capability of ALL_CAPABILITIES) expect(supports("pytorch", capability)).toBe(true);
    expect(missingCapabilities("pytorch")).toEqual([]);
  });

  it("the legacy hook engine only has layer activity", () => {
    expect(supports("transformers", "layerActivity")).toBe(true);
    expect(supports("transformers", "concepts")).toBe(false);
    expect(missingCapabilities("transformers")).not.toContain("layerActivity");
    expect(missingCapabilities("transformers")).toContain("safetyTrace");
  });

  it("a black-box adapter is missing everything", () => {
    expect(missingCapabilities("ollama")).toEqual(ALL_CAPABILITIES);
  });

  it("missing + supported always partitions the full set", () => {
    for (const name of ADAPTERS) {
      const have = ADAPTER_PROFILES[name].capabilities.length;
      expect(have + missingCapabilities(name).length).toBe(ALL_CAPABILITIES.length);
    }
  });
});

describe("profileFor", () => {
  it("falls back rather than returning undefined for an unknown adapter", () => {
    // An adapter added on the backend but not here must not crash the page.
    expect(profileFor("brand-new" as AdapterName)).toBeDefined();
    expect(profileFor("brand-new" as AdapterName).reasonKey).toBeTruthy();
  });
});
