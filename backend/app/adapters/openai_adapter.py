import time
from typing import AsyncGenerator
from openai import AsyncOpenAI
from app import refusal
from app.schemas import RunRequest


def _ev(type_: str, data: dict) -> dict:
    return {"type": type_, "ts": time.perf_counter(), "data": data}


class OpenaiAdapter:
    async def stream(self, request: RunRequest) -> AsyncGenerator[dict, None]:
        if not request.api_key:
            yield _ev("error", {"message": "OpenAI API key is missing. Please enter it in the settings."})
            return

        yield _ev("run_started", {"prompt_tokens": 0, "token_limit_mode": request.token_limit_mode, "requested_max_tokens": request.max_new_tokens, "effective_max_tokens": request.max_new_tokens, "context_length": None})

        full_text = request.assistant_prefill or ""
        try:
            client = AsyncOpenAI(api_key=request.api_key)
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.extend({"role": turn.role, "content": turn.content} for turn in request.history)
            messages.append({"role": "user", "content": request.prompt})
            if request.assistant_prefill:
                messages.append({"role": "assistant", "content": request.assistant_prefill})

            response = await client.chat.completions.create(
                model=request.model or "gpt-4o",
                messages=messages,
                max_tokens=request.max_new_tokens,
                temperature=request.temperature,
                stream=True,
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_text += text
                    yield _ev("token", {"text": text, "generated_text": full_text})

        except Exception as e:
            yield _ev("error", {"message": str(e)})

        assessment = refusal.assess_output(full_text, request.response_language)
        yield _ev("run_completed", {"generated_text": full_text, "refused": assessment["refusal_style"] in ("hard_refusal", "soft_refusal", "safe_redirect"), "outcome": assessment["legacy_outcome"], "assessment": assessment, "finish_reason": "stop", "state": "black_box", "errors": []})
