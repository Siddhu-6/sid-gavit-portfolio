"""LLM client wrappers.

Primary provider is Groq (OpenAI-compatible API, free tier, extremely fast
inference on GPT-OSS 120B). GeminiClient is kept below as a drop-in
alternative — just swap the import in agent.py.

Both clients expose:
    async generate(system_prompt, user_message) -> str           # non-streaming
    async generate_stream(system_prompt, user_message) -> AsyncIterator[str]  # streaming

Streaming is preferred for user-facing chat since it drops perceived latency
from ~1-2s (waiting for full response) to ~200-400ms (first token appears).
"""

import json
from typing import AsyncIterator, Optional

import httpx

from siddhikesh_agent.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)


class LLMError(Exception):
    """Raised for any recoverable error contacting the LLM."""


# ─── Shared httpx client (persistent connection pool to Groq) ──────────
# Reusing the client across requests skips TLS handshake and connection
# setup (~50-150ms savings per call). httpx.AsyncClient is thread/async safe
# and lazy — no cost until first use.
_HTTPX_CLIENT: Optional[httpx.AsyncClient] = None


def _get_httpx_client() -> httpx.AsyncClient:
    """Return the shared httpx client, creating it lazily on first use."""
    global _HTTPX_CLIENT
    if _HTTPX_CLIENT is None or _HTTPX_CLIENT.is_closed:
        _HTTPX_CLIENT = httpx.AsyncClient(
            timeout=LLM_TIMEOUT_SECONDS,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
        )
    return _HTTPX_CLIENT


class GroqClient:
    """Groq REST API client (via httpx).

    Groq uses an OpenAI-compatible /chat/completions endpoint. Their free tier
    gives 30 rpm / 14,400 rpd on GPT-OSS 120B with sub-second latency.
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set.")
        self.api_key = api_key
        self.model = model

    @classmethod
    def from_env(cls) -> "GroqClient":
        """Build a client from environment variables."""
        return cls(api_key=GROQ_API_KEY, model=GROQ_MODEL)

    def _build_payload(self, system_prompt: str, user_message: str,
                        stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "top_p": 0.95,
            "stream": stream,
            # GPT-OSS models are reasoning models by default. "low" prevents
            # them from burning output tokens on internal chain-of-thought
            # before producing the actual reply (Sid's voice = snappy, not essay).
            "reasoning_effort": "low",
        }

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _raise_for_status(self, status_code: int) -> None:
        if status_code == 401:
            raise LLMError("GROQ_API_KEY is invalid or missing.")
        if status_code == 413:
            raise LLMError("request payload too large.")
        if status_code == 429:
            raise LLMError("model rate-limited; try again in a moment.")
        if status_code >= 400:
            raise LLMError(f"model returned HTTP {status_code}.")

    async def generate(self, system_prompt: str, user_message: str) -> str:
        """Non-streaming: return the full reply text once complete."""
        url = f"{self.BASE_URL}/chat/completions"
        payload = self._build_payload(system_prompt, user_message, stream=False)
        client = _get_httpx_client()

        try:
            response = await client.post(url, json=payload, headers=self._headers())
        except httpx.TimeoutException as e:
            raise LLMError("request timed out.") from e
        except httpx.RequestError as e:
            raise LLMError(f"network error: {e}") from e

        self._raise_for_status(response.status_code)

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"unexpected model response: {e}") from e

    async def generate_stream(
        self, system_prompt: str, user_message: str
    ) -> AsyncIterator[str]:
        """Streaming: yield content chunks as Groq produces them via SSE.

        Groq returns Server-Sent Events with this format per chunk:
            data: {"choices":[{"delta":{"content":"hello"}}]}\\n\\n
            data: {"choices":[{"delta":{"content":" world"}}]}\\n\\n
            data: [DONE]\\n\\n

        We yield only the .content strings, in order.
        """
        url = f"{self.BASE_URL}/chat/completions"
        payload = self._build_payload(system_prompt, user_message, stream=True)
        client = _get_httpx_client()

        try:
            async with client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as response:
                self._raise_for_status(response.status_code)

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Skip malformed chunks — GPT-OSS reasoning-effort chunks
                        # may arrive as sparse deltas
                        continue
        except httpx.TimeoutException as e:
            raise LLMError("request timed out.") from e
        except httpx.RequestError as e:
            raise LLMError(f"network error: {e}") from e


class GeminiClient:
    """Google Gemini REST API client (via httpx). Kept as an alternative."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set.")
        self.api_key = api_key
        self.model = model

    @classmethod
    def from_env(cls) -> "GeminiClient":
        return cls(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)

    async def generate(self, system_prompt: str, user_message: str) -> str:
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": user_message}]},
            ],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": LLM_TEMPERATURE,
                "maxOutputTokens": LLM_MAX_TOKENS,
                "topP": 0.95,
                "topK": 40,
            },
            "safetySettings": [
                {"category": c, "threshold": "BLOCK_ONLY_HIGH"}
                for c in (
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                )
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
                response = await client.post(url, json=payload)
        except httpx.TimeoutException as e:
            raise LLMError("request timed out.") from e
        except httpx.RequestError as e:
            raise LLMError(f"network error: {e}") from e

        if response.status_code == 429:
            raise LLMError("model rate-limited; try again in a moment.")
        if not response.is_success:
            raise LLMError(f"model returned HTTP {response.status_code}.")

        try:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"unexpected model response: {e}") from e
