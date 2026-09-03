"""Loads and caches the knowledge base (about.md) as parsed sections.

Design:
- Parse about.md once per cold start into {section_title: section_content}
- On each request, build a compact KB by combining:
    (a) always-included CORE sections (identity, voice, rules, contact)
    (b) top-N sections matching keywords in the user's message

This keeps request payloads to Groq under ~4K tokens while preserving
the full richness of a ~34K-character about.md — solving the HTTP 413
we hit when the full KB was pushed on every request.
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

# Prefix-matched core sections (their real titles have parenthetical suffixes)
CORE_SECTION_PREFIXES = (
    "Signature phrases",       # voice contract
    "OUT-OF-CONTEXT HANDLING", # behavior rules
)

# ─── Topic sections + keywords that trigger them ───────────────────────
# Keys are matched as PREFIXES of actual section titles (which may have
# parenthetical suffixes in the source markdown).
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Personality — how to convey": ["tone", "voice", "how you talk",
                                      "how you speak", "how you sound"],
    "Personality quick-fire": ["prefer", " vs ", "versus", " or ",
                                "morning person", "night owl", "cats or dogs",
                                "introvert", "extrovert", "either or",
                                "which do you"],
    "Quick-answer lookup": ["why ai", "why ml", "why you", "hiring",
                              "salary", "available", "cpi", "dream company"],
    "Family": ["family", "parents", "mom", "mother", "dad", "father",
               "sister", "brother", "siblings", "home"],
    "Education": ["education", "school", "college", "cpi", "gpa", "grades",
                  "iiit", "vadodara", "12th", "class", "10th", "cbse",
                  "study", "studied", "student", "degree", "btech", "b.tech",
                  "drop year", "jee"],
    "Interests": ["hobby", "hobbies", "fun", "interest",
                   "spare time", "free time", "outside",
                   "coffee", "football", "reading", "music",
                   "travel", "cook", "gym", "language",
                   "night walk", "nature", "walk"],
    "Media & Culture": ["movie", "movies", "show", "shows", "series", "tv",
                        "netflix", "watch", "bollywood", "korean",
                        "running man", "walking dead", "zombie", "genre"],
    "Recent books read": ["book", "books", "read", "reading", "author",
                          "novel", "vex king", "manifest"],
    "Personal life": ["relationship", "love", "girlfriend",
                       "boyfriend", "dating", "marriage",
                       "personal life", " ex ", "single",
                       "believe in love"],
    "AI / tech origin story": ["why ai", "why ml", "why not", "how you got",
                                "started", "how did you get into",
                                "get into ai", "get into ml", "learn ai",
                                "learn ml"],
    "Origin story": ["origin", "how you got", "journey",
                      "how did you", "path", "story", "pivot"],
    "Projects": ["project", "projects", "built", "build",
                  "portfolio", "github", "shipped", "code",
                  "research assistant", "workflow", "churn",
                  "multi-agent", "langgraph", "agent"],
    "Experience": ["experience", "internship", "intern", "job", "research",
                    "work history", "teaching assistant", "ta ", "photographer",
                    "obscura", "worked", "current role"],
    "Achievements": ["achievement", "achievements", "hackathon", "hackathons",
                      "award", "won", "prize", "anveshan", "hackiiitv",
                      "adobe", "national", "state", "medal"],
    "Career direction & compensation": ["career", "salary", "compensation",
                                          "ctc", "lpa", "package", "hire",
                                          "hiring", "role", "position",
                                          "when can you start", "available",
                                          "availability", "off-campus",
                                          "off campus", "dream company",
                                          "target company", "startup",
                                          "company", "join"],
    "Work style & values": ["work style", "values", "how do you work",
                              "how you work", "ship", "shipping",
                              "production", "approach"],
    "Strengths": ["strength", "strengths", "strong", "good at",
                   "best at", "why you", "why should"],
    "Weaknesses": ["weakness", "weaknesses", "weak", "bad at",
                    "improve", "growth area"],
    "Hot takes / opinions": ["opinion", "opinions", "think about", "views",
                              "hot take", "controversial", "hype",
                              "overhyped", "underhyped", "believe"],
    "Daily routines": ["routine", "routines", "morning", "day", "daily",
                        "schedule", "typical day", "wake up", "sleep"],
    "What makes him happy": ["happy", "happiness", "joy", "enjoy",
                               "makes you happy"],
    "What makes him uncomfortable": ["uncomfortable", "hate", "dislike",
                                       "annoy", "annoyed", "annoys",
                                       "pet peeve"],
    "Life goal": ["goal in life", "goal", "goals", "future", "dream",
                   "aspiration", "long term", "5 years",
                   "five years", "vision", "purpose"],
    "Fun bits": ["fun bit", "fun bits", "random", "casual",
                  "guilty", "weird", "quirk", "quirks", "trivia",
                  "tell me something"],
}

# ─── Module-level cache ───────────────────────────────────────────────
_SECTIONS_CACHE: Optional[Dict[str, str]] = None
_RAW_CACHE: Optional[str] = None


def _parse_sections(markdown: str) -> Dict[str, str]:
    """Split about.md into {section_title: full_section_text_including_header}.

    Uses `#`, `##`, `###` markdown headers as section delimiters. Section
    boundaries end at the next same-or-higher-level header. The dict is
    keyed by the header text (without the leading # marks or trailing
    whitespace).
    """
    sections: Dict[str, str] = {}
    lines = markdown.split("\n")
    current_title: Optional[str] = None
    current_body: List[str] = []

    for line in lines:
        # Match top-level (#) or second-level (##) headers only —
        # deeper ### / #### stay part of their parent section.
        header_match = re.match(r"^(#{1,2})\s+(.+?)\s*$", line)
        if header_match:
            # Flush the previous section
            if current_title is not None:
                sections[current_title] = "\n".join(current_body).strip()
            # Start new section
            current_title = header_match.group(2).strip()
            current_body = [line]
        else:
            current_body.append(line)

    # Flush final section
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
    """Return the full knowledge base (for anything that wants everything).

    Kept for backward compatibility. New code should prefer
    `build_kb_for_message()` which selects a compact subset.
    """
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
    """Check if a section title is a core section (exact match or matches core prefix)."""
    if title in CORE_SECTIONS_EXACT:
        return True
    return any(title.startswith(prefix) for prefix in CORE_SECTION_PREFIXES)


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

    Always includes:
      - CORE_SECTIONS_EXACT + CORE_SECTION_PREFIXES-matched sections
        (identity, personality, voice, out-of-context rules, contact)
    Adds:
      - up to `max_extra_sections` topic sections whose keywords match the
        user's message, ranked by keyword-hit count

    Rationale: keeps total prompt payload well under Groq's per-request limit
    (~3K tokens vs. 8K+ if we included the full about.md),
    while ensuring the agent has the specific context needed for the current
    question. Also cushions against future KB growth — you can keep adding
    to about.md without ever risking a 413 again.
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

    # Highest score first, ties broken alphabetically (deterministic)
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked_extras = {title for _, title in scored[:max_extra_sections]}

    # Preserve about.md's original section order for natural narrative flow
    parts: List[str] = []
    for title, content in _SECTIONS_CACHE.items():
        if _is_core_section(title) or title in picked_extras:
            parts.append(content)

    return "\n\n---\n\n".join(parts)
