"""In-memory per-key rate limiter.

Uses a sliding fixed-window strategy: N requests per W seconds per key.
State lives in the process (dict), which means it resets across cold
starts. That's fine for portfolio-scale traffic — for real production,
back this with Redis.
"""

import time
from typing import Dict, Tuple

from siddhikesh_agent.config import (
    MAX_REQUESTS_PER_MINUTE,
    RATE_LIMIT_WINDOW_SECONDS,
)


class RateLimiter:
    """Simple fixed-window rate limiter keyed by arbitrary string (usually IP)."""

    def __init__(
        self,
        max_per_window: int = MAX_REQUESTS_PER_MINUTE,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
    ):
        self.max_per_window = max_per_window
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Tuple[int, float]] = {}

    def allow(self, key: str) -> bool:
        """Return True if the request should be allowed for this key."""
        now = time.time()
        count, reset_at = self._buckets.get(key, (0, 0.0))

        if now > reset_at:
            # window expired — start a new one
            self._buckets[key] = (1, now + self.window_seconds)
            return True

        if count >= self.max_per_window:
            return False

        self._buckets[key] = (count + 1, reset_at)
        return True

    def remaining(self, key: str) -> int:
        """Requests left in the current window (best-effort, doesn't consume one)."""
        now = time.time()
        count, reset_at = self._buckets.get(key, (0, 0.0))
        if now > reset_at:
            return self.max_per_window
        return max(0, self.max_per_window - count)
