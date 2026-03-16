from __future__ import annotations

import json
from pathlib import Path


def load_video_url_registry(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        return {}

    registry: dict[str, str] = {}
    for video_name, metadata in raw.items():
        if not isinstance(video_name, str):
            continue
        if not isinstance(metadata, dict):
            continue
        url = metadata.get("url")
        if isinstance(url, str) and url.strip():
            registry[video_name] = url.strip()
    return registry


def prettify_video_name(video_name: str) -> str:
    return " ".join(video_name.replace("_", " ").split())


def inject_video_urls(answer: str, registry: dict[str, str]) -> str:
    if not answer or not registry:
        return answer

    updated = answer
    for video_name in sorted(registry.keys(), key=len, reverse=True):
        url = registry[video_name]
        replacement = f"{prettify_video_name(video_name)} ({url})"
        updated = updated.replace(video_name, replacement)
    return updated
