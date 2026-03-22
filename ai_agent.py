import os
import re
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

# --- Module-level LLM (reused across requests) ---
_llm: Optional[ChatGroq] = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            temperature=0.55,
            model_name="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


# --- Tight system rules (latency + voice) ---
SYSTEM_PROMPT_BASE = """You are Kalmly, a calm AI wellness companion for students.
Keep ALL responses to a MAXIMUM of 1 sentence (2 short clauses at most).
Never exceed 20 words. No lists. No markdown. Plain text only.
Sound human and warm, not robotic. No emojis."""

# --- Emotion & crisis (fast heuristics) ---
_EMOTION_ORDER = [
    ("overwhelmed", [r"\boverwhelm", r"\btoo much\b", r"can't cope", r"drowning", r"burnout"]),
    ("anxious", [r"\banxious", r"\banxiety", r"\bpanic", r"\bworried", r"\bnervous"]),
    ("stressed", [r"\bstress", r"\bexam", r"\bdeadline", r"\bpressure", r"\bgrade"]),
    ("sad", [r"\bsad", r"\blonely", r"\bempty", r"\bcry", r"\bdepress"]),
]

_CRISIS_PATTERNS = [
    r"give\s+up",
    r"no\s+point",
    r"can't\s+do\s+this",
    r"can'?t\s+do\s+this",
    r"\buseless\b",
    r"\bworthless\b",
    r"\bhurt\s+myself\b",
    r"\bkill\s+myself\b",
    r"\bsuicid",
]


def detect_emotion(text: str) -> str:
    if not text or not text.strip():
        return "neutral"
    t = text.lower()
    for label, patterns in _EMOTION_ORDER:
        for p in patterns:
            if re.search(p, t):
                return label
    return "neutral"


def detect_crisis(text: str) -> bool:
    if not text or not text.strip():
        return False
    t = text.lower()
    for p in _CRISIS_PATTERNS:
        if re.search(p, t):
            return True
    return False


def build_system_prompt(emotion: str, crisis: bool, memory_summary: str) -> str:
    parts = [SYSTEM_PROMPT_BASE, f"The student seems: {emotion}."]
    if crisis:
        parts.append(
            "CRISIS: Acknowledge pain briefly. Ask if they are safe. "
            "Suggest a trusted person or crisis resources. Under 20 words total for your reply if possible."
        )
    if memory_summary.strip():
        parts.append(f"Memory: {memory_summary.strip()}")
    return "\n".join(parts)


def _strip_system(messages: list) -> list:
    return [m for m in (messages or []) if not isinstance(m, SystemMessage)]


def _chunk_text(chunk) -> str:
    c = getattr(chunk, "content", None)
    if c is None:
        return ""
    if isinstance(c, list):
        out = []
        for part in c:
            if isinstance(part, dict) and "text" in part:
                out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
            else:
                out.append(str(part))
        return "".join(out)
    return str(c)


async def stream_user_input(
    user_text: str,
    current_messages: list,
    memory_summary: str = "",
) -> AsyncGenerator[str, None]:
    """
    Stream LLM output; yield sentence-sized chunks at . ! ? … boundaries (then space or end).
    """
    emotion = detect_emotion(user_text)
    crisis = detect_crisis(user_text)
    system_content = build_system_prompt(emotion, crisis, memory_summary)

    history = _strip_system(current_messages)
    messages = [SystemMessage(content=system_content)] + history + [HumanMessage(content=user_text)]

    llm = _get_llm()
    buffer = ""

    async for chunk in llm.astream(messages):
        buffer += _chunk_text(chunk)
        # Sentence boundary: punctuation then whitespace or end of buffer (streaming end handled below)
        while True:
            m = re.search(r"(.+?[.!?…])(?=\s|$)", buffer, re.DOTALL)
            if not m:
                break
            sentence = m.group(1).strip()
            buffer = buffer[m.end() :].lstrip()
            if sentence:
                yield sentence

    tail = buffer.strip()
    if tail:
        yield tail


def merge_session_after_turn(
    current_messages: list,
    user_text: str,
    assistant_full_text: str,
) -> list:
    """Append Human + AI messages for the next turn (no LangGraph state object)."""
    base = _strip_system(current_messages or [])
    return base + [
        HumanMessage(content=user_text),
        AIMessage(content=assistant_full_text),
    ]


if __name__ == "__main__":
    import asyncio

    async def _demo():
        async for s in stream_user_input("I am stressed about exams.", [], ""):
            print("chunk:", s)

    asyncio.run(_demo())
