from __future__ import annotations

from typing import Iterable, Optional


DEFAULT_REASONING_TAG = "thinking"
REASONING_TAG_ALIASES = ("thinking", "think")


def normalize_reasoning_tag(reasoning_tag: Optional[str]) -> str:
    tag = reasoning_tag or DEFAULT_REASONING_TAG
    if tag not in REASONING_TAG_ALIASES:
        raise ValueError(f"Unsupported reasoning tag {tag!r}; expected one of {REASONING_TAG_ALIASES}")
    return tag


def tag_pair(tag: str) -> tuple[str, str]:
    normalized = normalize_reasoning_tag(tag)
    return f"<{normalized}>", f"</{normalized}>"


def reasoning_tag_order(reasoning_tag: Optional[str], include_aliases: bool = True) -> tuple[str, ...]:
    preferred = normalize_reasoning_tag(reasoning_tag)
    if not include_aliases:
        return (preferred,)
    return (preferred,) + tuple(tag for tag in REASONING_TAG_ALIASES if tag != preferred)


def render_reasoning_tags(text: str, reasoning_tag: Optional[str]) -> str:
    """Render literal reasoning delimiter tags in text with the configured tag name."""
    target = normalize_reasoning_tag(reasoning_tag)
    rendered = text
    for alias in REASONING_TAG_ALIASES:
        rendered = rendered.replace(f"<{alias}>", f"<{target}>")
        rendered = rendered.replace(f"</{alias}>", f"</{target}>")
    return rendered


def render_reasoning_tags_in_messages(messages: Iterable[dict], reasoning_tag: Optional[str]) -> list[dict]:
    rendered_messages = []
    for message in messages:
        rendered_message = message.copy()
        content = rendered_message.get("content")
        if isinstance(content, str):
            rendered_message["content"] = render_reasoning_tags(content, reasoning_tag)
        rendered_messages.append(rendered_message)
    return rendered_messages


def find_reasoning_pair(text: str, reasoning_tag: Optional[str]) -> tuple[str, int, int] | tuple[None, int, int]:
    """Find the first valid configured-or-legacy reasoning tag pair in text."""
    candidates: list[tuple[int, str, int, int]] = []
    for tag in reasoning_tag_order(reasoning_tag):
        open_tag, close_tag = tag_pair(tag)
        start = text.find(open_tag)
        end = text.find(close_tag, start + len(open_tag)) if start != -1 else -1
        if start != -1:
            candidates.append((start, tag, start, end))

    if not candidates:
        return None, -1, -1

    _, tag, start, end = min(candidates, key=lambda candidate: candidate[0])
    return tag, start, end
