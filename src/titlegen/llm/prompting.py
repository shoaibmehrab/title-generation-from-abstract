from __future__ import annotations

import re
from typing import Iterable, Tuple


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_title_prompt(
    abstract: str,
    instruction: str,
    few_shot_examples: Iterable[Tuple[str, str]],
) -> str:
    abstract = _normalize_ws(abstract)
    instruction = _normalize_ws(instruction)

    sections = [
        "You are an expert scientific writing assistant.",
        instruction,
        "Rules:",
        "- Return only one title.",
        "- Do not include explanations or prefixes.",
        "- Keep it concise and specific.",
    ]

    examples = list(few_shot_examples)
    if examples:
        sections.append("Examples:")
        for idx, (ex_abstract, ex_title) in enumerate(examples, start=1):
            sections.append(f"Example {idx} Abstract: {_normalize_ws(ex_abstract)}")
            sections.append(f"Example {idx} Title: {_normalize_ws(ex_title)}")

    sections.append(f"Abstract: {abstract}")
    sections.append("Title:")

    return "\n".join(sections)


def postprocess_title(raw_text: str, max_words: int = 15) -> str:
    text = _normalize_ws(raw_text)
    if not text:
        return ""

    # Keep first line only and strip simple wrappers.
    text = text.splitlines()[0].strip()
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text).strip()
    text = re.sub(r"^\s*title\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    words = text.split()
    if max_words > 0 and len(words) > max_words:
        text = " ".join(words[:max_words]).strip()

    return text