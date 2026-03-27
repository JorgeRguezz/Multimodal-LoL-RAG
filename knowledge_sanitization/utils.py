import json
import os
import re
import unicodedata
from difflib import get_close_matches
from typing import Any

META_TAG_RE = re.compile(r"<\|[^>]+\|>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
MULTI_WS_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{3,}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def strip_diacritics(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def clean_text(text: str, blocked_patterns: list[str] | None = None) -> tuple[str, dict]:
    if not isinstance(text, str):
        return "", {"non_string": 1}

    stats = {
        "meta_tags_removed": 0,
        "meta_patterns_removed": 0,
        "control_chars_removed": 0,
    }

    original = text
    text = normalize_unicode(text)

    meta_hits = len(META_TAG_RE.findall(text))
    if meta_hits:
        stats["meta_tags_removed"] = meta_hits
    text = META_TAG_RE.sub(" ", text)

    if blocked_patterns:
        for pat in blocked_patterns:
            new_text, n = re.subn(pat, " ", text, flags=re.IGNORECASE)
            text = new_text
            stats["meta_patterns_removed"] += n

    ctrl_hits = len(CONTROL_RE.findall(text))
    if ctrl_hits:
        stats["control_chars_removed"] = ctrl_hits
    text = CONTROL_RE.sub("", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTI_WS_RE.sub(" ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = MULTI_NL_RE.sub("\n\n", text)
    text = text.strip()

    if text.lower() in {"analysis", "final", "output"}:
        text = ""

    if not text and original.strip():
        stats["fully_removed"] = 1

    return text, stats


def parse_segment_time(value: str) -> tuple[float, float] | None:
    if not isinstance(value, str) or "-" not in value:
        return None
    left, right = value.split("-", 1)
    try:
        start = float(left.strip())
        end = float(right.strip())
    except ValueError:
        return None
    if end <= start:
        return None
    return (start, end)


def normalize_name(name: str, alias_map: dict[str, list[str]]) -> str:
    if not isinstance(name, str):
        return "Unknown"

    raw = normalize_unicode(name).strip().strip('"').strip("'")
    if not raw:
        return "Unknown"

    upper_raw = raw.upper()
    reverse = {}
    for canonical, aliases in alias_map.items():
        reverse[canonical.upper()] = canonical.upper()
        for alias in aliases:
            reverse[alias.upper()] = canonical.upper()

    if upper_raw in reverse:
        return reverse[upper_raw]

    ascii_upper = strip_diacritics(upper_raw)
    if ascii_upper in reverse:
        return reverse[ascii_upper]

    candidates = list(reverse.keys())
    match = get_close_matches(upper_raw, candidates, n=1, cutoff=0.92)
    if match:
        return reverse[match[0]]

    return upper_raw


def normalize_entity_type(value: str, allowed: set[str]) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    parts = [p.strip().upper().strip('"') for p in value.split("<SEP>") if p.strip()]
    for p in parts:
        if p in allowed:
            return p
    return "UNKNOWN"


def canonicalize_source_ids(source_id: str | list[str], valid_chunk_ids: set[str]) -> str:
    if isinstance(source_id, list):
        parts = source_id
    else:
        parts = str(source_id).split("<SEP>")
    cleaned = [p.strip() for p in parts if p and p.strip() in valid_chunk_ids]
    return "<SEP>".join(sorted(set(cleaned)))


def should_block_entity_name(name: str, blocked_placeholders: set[str]) -> bool:
    if not name:
        return True
    probe = name.strip().upper()
    if probe in blocked_placeholders:
        return True
    if probe.startswith("<") and probe.endswith(">"):
        return True
    if len(probe) > 120:
        return True
    return False
