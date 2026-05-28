import json
from typing import Optional

from openai import OpenAI

from recsys.config import MODEL_NAME, OPENROUTER_API_KEY, OPENROUTER_BASE_URL


class LLMClient:
    def __init__(self, model: Optional[str] = None):
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self.model = model or MODEL_NAME
        self._client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            reason = getattr(choice, "finish_reason", "unknown")
            raise RuntimeError(f"Empty response from model (finish_reason={reason})")
        return content

    def complete_json(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> dict:
        text = self.complete(
            messages, temperature=temperature, max_tokens=max_tokens, model=model
        ).strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        return json.loads(text)
