"""Environment configuration for sid.agent.

Every constant here comes from an environment variable (with a sensible
default) so nothing is hardcoded in the app logic. Change tuning without
touching business code.
"""

import os
from pathlib import Path

# ── Groq (primary — free tier, super fast) ──────────────
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# ── Google Gemini (alternative provider) ────────────────
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# ── Rate limiting (per client IP) ────────────────────────
MAX_REQUESTS_PER_MINUTE: int = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", "15"))
RATE_LIMIT_WINDOW_SECONDS: int = 60

# ── LLM generation tuning ────────────────────────────────
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.9"))
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "400"))
LLM_TIMEOUT_SECONDS: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "15"))

# ── Request validation ───────────────────────────────────
MAX_MESSAGE_LENGTH: int = 600

# ── Paths ────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_PATH: Path = PROJECT_ROOT / "about.md"
