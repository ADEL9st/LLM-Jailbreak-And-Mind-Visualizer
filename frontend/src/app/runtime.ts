import { assessmentForVisibleText } from "../outputAssessment";
import type { CompareResult, OutputAssessment, RunRequest } from "../types";

export const WS_URL = "ws://127.0.0.1:8000/ws/run";
export const API_BASE = "http://127.0.0.1:8000";

/** Run one prompt over an isolated WebSocket and collect its summary. */
export function runPromptWS(
  request: RunRequest,
  signal?: AbortSignal
): Promise<Omit<CompareResult, "mode" | "jailbreak">> {
  return new Promise((resolve) => {
    const startedAt = performance.now();
    let peak = 0;
    let state = "?";
    let refused: boolean | null = null;
    let text = "";
    const errors: string[] = [];
    let outcome: string | undefined;
    let assessment: OutputAssessment | undefined;
    let finishReason: string | undefined;
    let outputTokens: number | undefined;
    let coherent: boolean | undefined;
    let settled = false;
    const socket = new WebSocket(WS_URL);

    const finish = () => {
      if (settled) return;
      settled = true;
      resolve({
        peak,
        state,
        refused,
        text,
        errors,
        elapsed: (performance.now() - startedAt) / 1000,
        outcome,
        assessment,
        finish_reason: finishReason,
        output_tokens: outputTokens,
        coherent
      });
    };

    if (signal) {
      signal.addEventListener("abort", () => {
        if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "stop" }));
        window.setTimeout(() => socket.close(), 1500);
      });
    }

    socket.onopen = () => socket.send(JSON.stringify(request));
    socket.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data) as { type: string; data: Record<string, unknown> };
        if (event.type === "safety_trace") {
          const data = event.data as { score: number; state: string };
          if (data.score > peak) {
            peak = data.score;
            state = data.state;
          }
        }
        if (event.type === "run_completed") {
          const data = event.data as {
            refused?: boolean;
            generated_text?: string;
            outcome?: string;
            assessment?: OutputAssessment;
            finish_reason?: string;
            output_tokens?: number;
            coherent?: boolean;
          };
          text = (data.generated_text ?? "").trim();
          assessment = assessmentForVisibleText(data.assessment, text) ?? undefined;
          refused = text ? data.refused ?? null : null;
          outcome = text ? data.outcome : "empty";
          finishReason = data.finish_reason;
          outputTokens = data.output_tokens;
          coherent = data.coherent;
        }
        if (event.type === "error") {
          errors.push(String((event.data as { message?: string }).message ?? event.data));
        }
      } catch {
        // Ignore malformed frames; the socket can continue streaming.
      }
    };
    socket.onerror = () => errors.push("websocket error");
    socket.onclose = finish;
  });
}
