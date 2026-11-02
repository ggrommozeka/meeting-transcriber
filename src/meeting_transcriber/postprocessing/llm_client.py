"""Minimal OpenAI-compatible chat client for a local LLM."""

from __future__ import annotations

import os

import requests

from ..config import LLMConfig
from ..errors import PipelineError


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        base = cfg.base_url.rstrip("/")
        self.endpoint = base + "/chat/completions"

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        api_key = os.environ.get(self.cfg.api_key_env, "not-needed")
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=self.cfg.timeout,
            )
        except requests.RequestException as exc:
            raise PipelineError(
                f"local LLM request failed ({self.endpoint}): {exc}. "
                "Is LM Studio / llama.cpp server running?"
            ) from exc
        if response.status_code >= 400:
            raise PipelineError(
                f"local LLM returned HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise PipelineError(f"unexpected local LLM response: {response.text[:500]}") from exc

    def unload(self) -> None:
        """No-op: the llama.cpp server owns the model lifetime.

        Use ``scripts/stop_llama_server.sh`` to release VRAM. Keeping the model
        resident makes repeated ``report`` runs fast.
        """

        return
