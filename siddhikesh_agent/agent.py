"""The sid.agent — orchestrates knowledge base + LLM into a chat agent.

Public API:
    agent = Agent.from_env()                # cold-start setup
    reply = await agent.chat(message)       # non-streaming
    async for chunk in agent.chat_stream(message):   # streaming
        ...

Also exposes a module-level get_agent() factory that caches the Agent
instance across warm invocations — Vercel serverless functions keep the
Python process alive between requests, so skipping re-init saves 20-50ms
per warm request.

Design note: the LLM client is cached, but the system prompt is built
PER REQUEST — selecting only KB sections relevant to the user's message.
This keeps request payloads under Groq's per-request token limit while
preserving KB richness.
"""

from typing import AsyncIterator, Optional

from siddhikesh_agent.config import MAX_MESSAGE_LENGTH
from siddhikesh_agent.knowledge import build_kb_for_message
from siddhikesh_agent.llm import GroqClient
from siddhikesh_agent.prompts import build_system_prompt


class Agent:
    """LLM-backed agent that answers visitor questions about Sid."""

    def __init__(self, llm: GroqClient):
        self.llm = llm

    @classmethod
    def from_env(cls) -> "Agent":
        """Cold-start factory: build LLM client from env vars."""
        llm = GroqClient.from_env()
        return cls(llm=llm)

    def _validate(self, message: str) -> str:
        """Strip + validate the incoming message. Raises ValueError."""
        message = (message or "").strip()
        if not message:
            raise ValueError("message is empty.")
        if len(message) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"message too long ({len(message)} chars). "
                f"keep it under {MAX_MESSAGE_LENGTH}."
            )
        return message

    async def chat(self, message: str) -> str:
        """
        Non-streaming: answer with a 1-3 sentence reply in Sid's voice.
        Returns the full reply once the LLM finishes generating.

        Raises:
            ValueError: if the message is empty or exceeds MAX_MESSAGE_LENGTH.
            LLMError:   if the LLM call fails (network, rate limit, format).
        """
        message = self._validate(message)
        kb = build_kb_for_message(message)
        system_prompt = build_system_prompt(kb)

        reply = await self.llm.generate(
            system_prompt=system_prompt,
            user_message=message,
        )
        return reply or "hmm, didn't quite catch that — try rephrasing?"

    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        """
        Streaming: yield reply chunks as the LLM produces them.

        First token typically arrives 200-400ms after the call starts,
        vs. 800-2000ms for the non-streaming version. This makes the
        agent feel dramatically snappier to end users.

        Raises:
            ValueError: if the message is empty or exceeds MAX_MESSAGE_LENGTH.
            LLMError:   if the LLM call fails.
        """
        message = self._validate(message)
        kb = build_kb_for_message(message)
        system_prompt = build_system_prompt(kb)

        async for chunk in self.llm.generate_stream(
            system_prompt=system_prompt,
            user_message=message,
        ):
            yield chunk


# ─── Module-level cached agent ─────────────────────────────────────────
# Vercel keeps the Python process warm between requests for a while.
# Caching the agent at module level means warm invocations skip:
#   - GroqClient construction
#   - Environment variable re-reading
# Savings: ~20-50ms per warm request.
_AGENT_CACHE: Optional[Agent] = None


def get_agent() -> Agent:
    """Return a shared Agent instance, initializing on first use."""
    global _AGENT_CACHE
    if _AGENT_CACHE is None:
        _AGENT_CACHE = Agent.from_env()
    return _AGENT_CACHE
