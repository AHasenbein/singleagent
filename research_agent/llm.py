"""Thin wrapper around the DeepSeek chat API (OpenAI-compatible).

Pipeline stages depend on the LLM protocol below, not on DeepSeek directly, so
a different provider or a test fake can be swapped in.
"""

import json
from typing import Optional, Protocol

from .config import DEEPSEEK_BASE_URL, DEFAULT_MODEL


class LLM(Protocol):
    def complete_json(self, system_prompt: str, user_text: str) -> dict:
        """Return the model's reply parsed as a JSON object."""

    def complete_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Return the model's plain-text reply to a single user prompt."""


def parse_json_object(text: str) -> dict:
    """Parse the single JSON object in a model reply.

    Code fences and leading prose before the first "{" are tolerated. Anything
    other than whitespace after the object (a second object, trailing text) is
    rejected: `{"status": "PASS"} {"status": "FAIL"}` must not read as PASS.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    start = text.find("{")
    if start == -1:
        raise ValueError("Model reply contained no JSON object.")
    value, end = json.JSONDecoder().raw_decode(text, start)
    if text[end:].strip():
        raise ValueError("Model reply had content after the JSON object.")
    if not isinstance(value, dict):
        raise ValueError("Model reply was not a JSON object.")
    return value


def _message_content(response) -> str:
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Model returned no content.")
    return content


class DeepSeekLLM:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = DEEPSEEK_BASE_URL):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete_json(self, system_prompt: str, user_text: str, attempts: int = 2) -> dict:
        """JSON reply; retries once if the model returns unparseable output."""
        for attempt in range(attempts):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                response_format={"type": "json_object"},
            )
            try:
                return parse_json_object(_message_content(response))
            except ValueError:  # json.JSONDecodeError is a ValueError
                if attempt == attempts - 1:
                    raise

    def complete_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return _message_content(response)
