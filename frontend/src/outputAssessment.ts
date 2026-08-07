import type { OutputAssessment } from "./types";

/**
 * Keep the UI honest when a stale backend reports success for an empty answer.
 * The backend performs the same check; this fallback protects saved runs and
 * older servers that may still send the previous assessment shape.
 */
export function assessmentForVisibleText(
  assessment: OutputAssessment | null | undefined,
  text: string
): OutputAssessment | null {
  if (!assessment) return null;
  if (text.trim()) return assessment;

  return {
    ...assessment,
    category: "empty",
    content_status: "empty",
    complete: false,
    coherent: false,
    manual_review_required: true,
    legacy_outcome: "empty"
  };
}
