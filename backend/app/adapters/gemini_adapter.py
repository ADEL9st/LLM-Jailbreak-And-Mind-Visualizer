import time
from typing import AsyncGenerator
from google import genai
from app import refusal
from app.schemas import RunRequest


def _ev(type_: str, data: dict) -> dict:
    return {"type": type_, "ts": time.perf_counter(), "data": data}


class GeminiAdapter:
    async def stream(self, request: RunRequest) -> AsyncGenerator[dict, None]:
        if not request.api_key:
            yield _ev("error", {"message": "Gemini API key is missing. Please enter it in the settings."})
            return

        yield _ev("run_started", {"prompt_tokens": 0, "token_limit_mode": request.token_limit_mode, "requested_max_tokens": request.max_new_tokens, "effective_max_tokens": request.max_new_tokens, "context_length": None})

        full_text = request.assistant_prefill or ""
        try:
            client = genai.Client(api_key=request.api_key)

            contents = []
            for turn in request.history:
                role = "user" if turn.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn.content}]})
            contents.append({"role": "user", "parts": [{"text": request.prompt}]})
            if request.assistant_prefill:
                contents.append({"role": "model", "parts": [{"text": request.assistant_prefill}]})

            response = await client.aio.models.generate_content_stream(
                model=request.model or "gemini-1.5-pro",
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    system_instruction=request.system_prompt or None,
                ),
            )

            async for chunk in response:
                if chunk.text:
                    text = chunk.text
                    full_text += text
                    yield _ev("token", {"text": text, "generated_text": full_text})

        except Exception as e:
            yield _ev("error", {"message": str(e)})

        assessment = refusal.assess_output(full_text, request.response_language)
        yield _ev("run_completed", {"generated_text": full_text, "refused": assessment["refusal_style"] in ("hard_refusal", "soft_refusal", "safe_redirect"), "outcome": assessment["legacy_outcome"], "assessment": assessment, "finish_reason": "stop", "state": "black_box", "errors": []})
