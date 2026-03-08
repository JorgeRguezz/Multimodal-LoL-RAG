"""
Sequential runner for knowledge build over sanitized_extracted_data folders.
"""

import argparse
import asyncio
import os
import sys
from typing import List, Tuple

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_build.builder import KnowledgeBuilder


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_extraction_dir() -> str:
    return os.path.join(_project_root(), "knowledge_sanitization", "cache")


def _discover_candidates(extraction_dir: str) -> List[str]:
    extracted_data_dir = os.path.join(extraction_dir, "sanitized_extracted_data")
    if not os.path.isdir(extracted_data_dir):
        return []

    candidates = []
    for entry in os.scandir(extracted_data_dir):
        if entry.is_dir():
            candidates.append(entry.path)
    return sorted(candidates)


def _is_no_unbuilt_error(exc: Exception) -> bool:
    if not isinstance(exc, FileNotFoundError):
        return False
    return "No unbuilt extraction folders found" in str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run knowledge_build.builder sequentially for sanitized_extracted_data folders "
            "until no unbuilt folders remain."
        )
    )
    parser.add_argument(
        "--extraction-dir",
        default=_default_extraction_dir(),
        help="Directory containing sanitized_extracted_data/ from knowledge_sanitization.",
    )
    args = parser.parse_args()

    extraction_dir = os.path.abspath(args.extraction_dir)
    if not os.path.isdir(extraction_dir):
        print(f"Extraction directory not found: {extraction_dir}")
        return 1

    candidates = _discover_candidates(extraction_dir)
    data_dir = os.path.join(extraction_dir, "sanitized_extracted_data")
    if not candidates:
        print(f"No sanitized_extracted_data folders found in: {data_dir}")
        return 1

    print(f"Found {len(candidates)} sanitized folder(s) in {data_dir}")

    processed_count = 0
    failures: List[Tuple[str, int, str]] = []

    while True:
        try:
            builder = KnowledgeBuilder(extraction_dir=extraction_dir)
        except Exception as exc:
            if _is_no_unbuilt_error(exc):
                break
            print(f"Failed to initialize builder: {exc}")
            return 1

        processed_count += 1
        artifact_name = os.path.basename(builder.artifact_dir)
        print(f"\n[{processed_count}] Building: {artifact_name}")
        print(f"Artifact dir: {builder.artifact_dir}")
        print(f"Output dir:   {builder.working_dir}")

        try:
            asyncio.run(builder.build())
            print(f"Completed: {artifact_name}")
        except Exception as exc:
            failures.append((artifact_name, 1, str(exc)))
            print(f"Failed: {artifact_name}")
            print(f"Reason: {exc}")
            # Stop on first failure to avoid infinite retries on the same folder.
            break

    print("\nBuild queue complete.")
    if failures:
        print(f"Failures: {len(failures)}")
        for name, code, reason in failures:
            print(f" - [{code}] {name}: {reason}")
        return 1

    if processed_count == 0:
        print("No unbuilt folders remained.")
    else:
        print(f"Processed {processed_count} folder(s) successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
