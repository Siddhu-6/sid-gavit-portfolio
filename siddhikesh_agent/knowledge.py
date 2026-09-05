"""Loads and caches the knowledge base (about.md) as parsed sections.

Design:
- Parse about.md once per cold start into {section_title: section_content}
- On each request, build a compact KB by combining:
    (a) always-included CORE sections (identity, voice, rules, contact, quick-answers)
    (b) top-N sections matching keywords in the user's message
    (c) FALLBACK: if no topic sections matched, include Interests + Fun bits
        so casual questions never come back empty

This keeps request payloads to Groq under ~4-5K tokens while preserving
the full richness of a ~34K-character about.md.
"""

import re
from typing import Dict, List, Optional, Tuple

from siddhikesh_agent.config import KNOWLEDGE_BASE_PATH


# ─── Sections that ship with EVERY request (identity + rules + voice) ──
# Uses exact match for these — keeps the core lean.
CORE_SECTIONS_EXACT = {
    "About Siddhikesh Arvind Gavit",  # top-of-file preamble
    "Identity",
    "Personality",
    "Contact",
}

# Prefix-matched core sections (real titles have parenthetical suffixes)
CORE_SECTION_PREFIXES = (
    "Signature phrases",        # voice contract
    "OUT-OF-CONTEXT HANDLING",  # behavior rules
    "Quick-answer lookup",      # authoritative fallback answers for common Qs
)

# ─── Fallback sections (loaded ONLY when no topic keywords matched) ───
# Ensures casual/personal questions phrased in unexpected ways still get
# rich context. E.g. "fav player and tactics" doesn't contain "football",
# but the Interests section has the football → Messi info.
FALLBACK_SECTION_PREFIXES = (
    "Interests",   # long-form section with football, coffee, music, etc.
    "Fun bits",    # grab bag of casual details
)

# ─── Topic sections + keywords that trigger them ───────────────────────
# Keys are matched as PREFIXES of actual section titles (which may have
# parenthetical suffixes in the source markdown).
#
# Keywords are exhaustive — every natural phrasing a user might use to
# ask about a topic. Better to over-match than under-match; irrelevant
# sections lose to more-relevant ones on score.
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Personality — how to convey": [
        "tone", "voice", "how you talk", "how you speak", "how you sound",
        "personality trait", "vibe", "character", "attitude",
    ],
    "Personality quick-fire": [
        "prefer", " vs ", "versus", " or ", "morning person", "night owl",
        "cats or dogs", "introvert", "extrovert", "either or",
        "which do you", "would you rather",
    ],
    "Quick-answer lookup": [
        # Broad — this section is designed as a fallback for common Qs
        "why ai", "why ml", "why you", "hiring", "salary", "available",
        "cpi", "dream company", "different", "unique", "stand out",
        "makes you", "why should we hire",
    ],
    "Family": [
        "family", "parents", "mom", "mother", "dad", "father",
        "sister", "brother", "siblings", "home", "background",
        "grew up", "childhood",
    ],
    "Education": [
        "education", "school", "college", "cpi", "gpa", "grades",
        "iiit", "vadodara", "12th", "class", "10th", "cbse",
        "study", "studied", "student", "degree", "btech", "b.tech",
        "drop year", "jee", "academic", "university", "institute",
    ],
    "Interests": [
        # Hobby-general
        "hobby", "hobbies", "fun", "interest", "spare time", "free time",
        "outside", "downtime", "unwind", "relax", "leisure", "pastime",
        # Football-specific
        "football", "soccer", "player", "players", "team", "match", "matches",
        "position", "positions", "tactic", "tactics", "style of play",
        "playing", "play", "coaching", "coach", "drill", "drills",
        "technique", "techniques", "approach", "winning", "win", "goal",
        "striker", "forward", "midfielder", "defender", "training",
        "sport", "sports", "fitness", "messi", "ronaldo",
        # Coffee-specific
        "coffee", "brew", "brewing", "espresso", "pour", "pour-over",
        "filter", "black", "cafe", "café", "beans", "roast",
        # Music-specific
        "music", "song", "songs", "band", "artist", "genre", "playlist",
        "listen", "lo-fi", "bollywood",
        # Reading
        "read", "reading", "book", "books", "novel", "author",
        # Travel
        "travel", "trip", "vacation", "destination", "country", "countries",
        "japan", "korea", "switzerland",
        # Cooking
        "cook", "cooking", "food", "eat", "cuisine", "dish", "meal", "recipe",
        # Gym
        "gym", "workout", "lift", "exercise", "cardio", "strength",
        # Languages
        "language", "languages", "spanish", "korean", "urdu", "learning language",
        # Nature
        "night walk", "nature", "walk", "walks", "stargazing", "stars",
        "river", "rivers", "mountain", "mountains", "beach", "solitude",
        "outdoors", "hiking",
    ],
    "Media & Culture": [
        "movie", "movies", "film", "films", "show", "shows", "series",
        "tv", "netflix", "watch", "watching", "bollywood", "korean",
        "running man", "walking dead", "zombie", "genre", "documentary",
        "ghibli", "anime",
    ],
    "Recent books read": [
        "book", "books", "read", "reading", "author", "novel",
        "vex king", "manifest", "alchemist", "reading list",
    ],
    "Personal life": [
        "relationship", "love", "girlfriend", "boyfriend", "dating",
        "marriage", "personal life", " ex ", "single", "believe in love",
        "romantic", "partner", "crush",
    ],
    "AI / tech origin story": [
        "why ai", "why ml", "why not", "how you got", "started",
        "how did you get into", "get into ai", "get into ml",
        "learn ai", "learn ml", "beginning", "how it started",
    ],
    "Origin story": [
        "origin", "how you got", "journey", "how did you", "path",
        "story", "pivot", "your story", "background story",
    ],
    "Projects": [
        "project", "projects", "built", "build", "building",
        "portfolio", "github", "shipped", "code", "coding",
        "research assistant", "workflow", "churn", "multi-agent",
        "langgraph", "agent", "what have you built", "what did you build",
        "your work", "sample work", "showcase", "demo",
    ],
    "Experience": [
        "experience", "internship", "intern", "job", "research",
        "work history", "teaching assistant", "ta ", "photographer",
        "obscura", "worked", "current role", "past role", "resume",
        "cv", "background", "professional",
    ],
    "Achievements": [
        "achievement", "achievements", "hackathon", "hackathons",
        "award", "won", "prize", "anveshan", "hackiiitv",
        "adobe", "national", "state", "medal", "recognition",
        "accolade",
    ],
    "Career direction & compensation": [
        "career", "salary", "compensation", "ctc", "lpa", "package",
        "hire", "hiring", "role", "position", "when can you start",
        "available", "availability", "off-campus", "off campus",
        "dream company", "target company", "startup", "company",
        "join", "notice period", "onboarding", "location",
    ],
    "Work style & values": [
        "work style", "values", "how do you work", "how you work",
        "ship", "shipping", "production", "approach to work",
        "philosophy", "principle", "principles",
    ],
    "Strengths": [
        "strength", "strengths", "strong", "good at", "best at",
        "why you", "why should", "superpower", "advantage",
    ],
    "Weaknesses": [
        "weakness", "weaknesses", "weak", "bad at", "improve",
        "growth area", "areas of growth", "development area",
        "shortcoming", "limitation",
    ],
    "Hot takes / opinions": [
        "opinion", "opinions", "think about", "views", "hot take",
        "controversial", "hype", "overhyped", "underhyped", "believe",
        "thoughts on", "what do you think", "unpopular",
    ],
    "Daily routines": [
        "routine", "routines", "morning", "day", "daily",
        "schedule", "typical day", "wake up", "sleep",
        "when do you", "how do you spend",
    ],
    "What makes him happy": [
        "happy", "happiness", "joy", "enjoy", "makes you happy",
        "brings you joy", "love doing",
    ],
    "What makes him uncomfortable": [
        "uncomfortable", "hate", "dislike", "annoy", "annoyed",
        "annoys", "pet peeve", "irritate", "can't stand",
    ],
    "Life goal": [
        "goal in life", "goal", "goals", "future", "dream",
        "aspiration", "long term", "5 years", "five years",
        "vision", "purpose", "life plan", "ambition",
    ],
    "Fun bits": [
        "fun bit", "fun bits", "random", "casual", "guilty",
        "weird", "quirk", "quirks", "trivia", "tell me something",
        "surprise me", "fun fact", "did you know",
    ],
}

# ─── Module-level cache ───────────────────────────────────────────────
_SECTIONS_CACHE: Optional[Dict[str, str]] = None
_RAW_CACHE: Optional[str] = None


def _parse_sections(markdown: str) -> Dict[str, str]:
    """Split about.md into {section_title: full_section_text_including_header}.

    Uses `#` and `##` markdown headers as section delimiters. Deeper `###`
    stays within its parent section.
    """
    sections: Dict[str, str] = {}
    lines = markdown.split("\n")
    current_title: Optional[str] = None
    current_body: List[str] = []

    for line in lines:
        header_match = re.match(r"^(#{1,2})\s+(.+?)\s*$", line)
        if header_match:
            if current_title is not None:
                sections[current_title] = "\n".join(current_body).strip()
            current_title = header_match.group(2).strip()
            current_body = [line]
        else:
            current_body.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(current_body).strip()

    return sections


def _load_from_disk() -> None:
    """Load and parse about.md into cache."""
    global _SECTIONS_CACHE, _RAW_CACHE
    try:
        _RAW_CACHE = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
        _SECTIONS_CACHE = _parse_sections(_RAW_CACHE)
    except FileNotFoundError:
        _RAW_CACHE = "Knowledge base file not found."
        _SECTIONS_CACHE = {"fallback": _RAW_CACHE}
    except OSError as e:
        _RAW_CACHE = f"Knowledge base unreadable: {e}"
        _SECTIONS_CACHE = {"fallback": _RAW_CACHE}


def load_knowledge() -> str:
    """Return the full knowledge base (for anything that wants everything)."""
    if _RAW_CACHE is None:
        _load_from_disk()
    return _RAW_CACHE or ""


def reload_knowledge() -> str:
    """Force a fresh read from disk (useful during local dev)."""
    global _SECTIONS_CACHE, _RAW_CACHE
    _SECTIONS_CACHE = None
    _RAW_CACHE = None
    return load_knowledge()


def _is_core_section(title: str) -> bool:
    """Check if a section title is a core section (exact or prefix match)."""
    if title in CORE_SECTIONS_EXACT:
        return True
    return any(title.startswith(prefix) for prefix in CORE_SECTION_PREFIXES)


def _is_fallback_section(title: str) -> bool:
    """Check if a section title is one of the fallback sections."""
    return any(title.startswith(prefix) for prefix in FALLBACK_SECTION_PREFIXES)


def _score_section(section_title: str, query_lower: str) -> int:
    """Count keyword hits between the user's query and a section's keyword list.

    Section titles are matched by prefix, since the actual titles in about.md
    often have parenthetical suffixes like '(for the interview question)'.
    """
    for prefix, keywords in TOPIC_KEYWORDS.items():
        if section_title.startswith(prefix):
            return sum(1 for kw in keywords if kw in query_lower)
    return 0


def build_kb_for_message(user_message: str, max_extra_sections: int = 3) -> str:
    """Assemble a compact KB tailored to the current user message.

    Loading strategy:
      1. ALWAYS include core sections (identity, voice, rules, contact,
         quick-answers)
      2. Score all other sections by keyword overlap with the message
      3. If any topic sections scored > 0: include top-N (default 3)
      4. If NO topic sections scored: fall back to Interests + Fun bits
         so casual questions always have rich hobby/personality material
         to draw from

    Rationale for the fallback: keyword matching is brittle — users don't
    always use the words we anticipated. E.g. "fav player and tactics"
    doesn't contain "football" but the Interests section is exactly what
    the LLM needs to answer accurately.
    """
    if _SECTIONS_CACHE is None:
        _load_from_disk()
    assert _SECTIONS_CACHE is not None  # for the type checker

    query_lower = " " + user_message.lower() + " "  # padded so " ex " / " or " match

    # Score all non-core sections by keyword overlap
    scored: List[Tuple[int, str]] = []
    for title in _SECTIONS_CACHE:
        if _is_core_section(title):
            continue
        score = _score_section(title, query_lower)
        if score > 0:
            scored.append((score, title))

    # Pick top-N by score (ties broken alphabetically for determinism)
    if scored:
        scored.sort(key=lambda x: (-x[0], x[1]))
        picked_extras = {title for _, title in scored[:max_extra_sections]}
    else:
        # FALLBACK: no keyword matched — load safe defaults for casual Qs
        picked_extras = {
            title for title in _SECTIONS_CACHE
            if _is_fallback_section(title)
        }

    # Preserve about.md's original section order for natural narrative flow
    parts: List[str] = []
    for title, content in _SECTIONS_CACHE.items():
        if _is_core_section(title) or title in picked_extras:
            parts.append(content)

    return "\n\n---\n\n".join(parts)