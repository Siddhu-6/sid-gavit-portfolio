"""siddhikesh_agent — the AI agent powering Sid's portfolio.

Public API:
    from siddhikesh_agent import Agent, LLMError, RateLimiter
"""

from siddhikesh_agent.agent import Agent
from siddhikesh_agent.llm import GeminiClient, LLMError
from siddhikesh_agent.ratelimit import RateLimiter

__all__ = ["Agent", "GeminiClient", "LLMError", "RateLimiter"]
__version__ = "2.0.0"
