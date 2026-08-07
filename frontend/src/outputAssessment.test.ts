import { describe, expect, it } from "vitest";
import { assessmentForVisibleText } from "./outputAssessment";
import type { OutputAssessment } from "./types";

const reportedSuccess: OutputAssessment = {
  category: "complete_compliance",
  content_status: "complete_compliance",
  refusal_style: "none",
  truncated: false,
  complete: true,
  coherent: true,
  manual_review_required: true,
  legacy_outcome: "compliance"
};

describe("assessmentForVisibleText", () => {
  it("reclassifies an empty answer instead of displaying a false success", () => {
    expect(assessmentForVisibleText(reportedSuccess, "   ")).toMatchObject({
      category: "empty",
      content_status: "empty",
      complete: false,
      coherent: false,
      legacy_outcome: "empty"
    });
  });

  it("keeps the backend assessment when the answer is visible", () => {
    expect(assessmentForVisibleText(reportedSuccess, "Visible answer")).toBe(reportedSuccess);
  });

  it("does not invent an assessment when the backend omitted it", () => {
    expect(assessmentForVisibleText(undefined, "")).toBeNull();
  });
});
