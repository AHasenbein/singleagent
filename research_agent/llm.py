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


class DeepSeekLLM:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = DEEPSEEK_BASE_URL):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete_json(self, system_prompt: str, user_text: str) -> dict:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def complete_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content
