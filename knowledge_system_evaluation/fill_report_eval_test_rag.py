from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_inference import InferenceService


DEFAULT_INPUT = Path(__file__).resolve().parent / "merged_report_eval_test.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "report_eval_test_rag_merged.json"


def _render_progress(current: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[no work]"
    ratio = min(1.0, max(0.0, current / total))
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio * 100:5.1f}%)"


def _print_progress(current: int, total: int) -> None:
    end = "\n" if current >= total else "\r"
    print(_render_progress(current, total), end=end, flush=True)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON list in {path}")
    return data


def _write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=4, ensure_ascii=True)
        f.write("\n")


def run(input_path: Path, output_path: Path, limit: int | None = None) -> None:
    cases = _load_cases(input_path)
    service = InferenceService()

    result_cases: list[dict[str, Any]] = []
    total = len(cases) if limit is None else min(len(cases), max(0, limit))
    _print_progress(0, total)

    for idx, case in enumerate(cases[:total], start=1):
        item = dict(case)
        question = str(item.get("question", "")).strip()

        if not question:
            item["rag_answer"] = ""
            item["context"] = ""
            result_cases.append(item)
            _print_progress(idx, total)
            continue

        result = service.answer(question)
        item["rag_answer"] = result.answer
        item["context"] = result.context
        result_cases.append(item)

        _write_cases(output_path, result_cases)
        _print_progress(idx, total)

    if total < len(cases):
        result_cases.extend(cases[total:])
        _write_cases(output_path, result_cases)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy report_eval_test.json and fill rag_answer/context via knowledge inference."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source evaluation JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination JSON copy")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N objects")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(input_path=args.input.resolve(), output_path=args.output.resolve(), limit=args.limit)


if __name__ == "__main__":
    main()
