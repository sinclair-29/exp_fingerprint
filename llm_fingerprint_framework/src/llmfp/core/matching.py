from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_normalized_target(output: str, target: str) -> bool:
    normalized_output = f" {normalize_text(output)} "
    normalized_target = normalize_text(target)
    if not normalized_target:
        return False
    return f" {normalized_target} " in normalized_output


def exact_normalized_target(output: str, target: str) -> bool:
    normalized_target = normalize_text(target)
    if not normalized_target:
        return False
    return normalize_text(output) == normalized_target


def parse_digit_sequence(output: str, n_digits: int) -> str | None:
    for match in re.finditer(r"\d+", output or ""):
        token = match.group(0)
        if len(token) == n_digits:
            return token
    return None


def match_trap_output(output: str, target_digits: str) -> tuple[bool, bool, str | None]:
    parsed = parse_digit_sequence(output, len(target_digits))
    invalid = parsed is None
    return parsed == target_digits, invalid, parsed


def match_keyword_target(output: str, target: str, target_keywords: list[str] | None = None) -> bool:
    if contains_normalized_target(output, target):
        return True
    for keyword in target_keywords or []:
        if contains_normalized_target(output, keyword):
            return True
    return False
