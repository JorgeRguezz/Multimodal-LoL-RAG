"""
Sequential runner for knowledge extraction over all videos in downloads/.
"""
import argparse
import os
import subprocess
import sys
from typing import List


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_extraction_dir() -> str:
    return os.path.join(_project_root(), "knowledge_extraction", "cache")


def _discover_videos(downloads_dir: str) -> List[str]:
    videos = []
    for entry in os.scandir(downloads_dir):
        if not entry.is_file():
            continue
        ext = os.path.splitext(entry.name)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            videos.append(entry.path)
    return sorted(videos)


def _expected_extracted_dir(extraction_dir: str, video_path: str) -> str:
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(extraction_dir, "extracted_data", video_basename)


def _is_video_already_extracted(extraction_dir: str, video_path: str) -> bool:
    extracted_dir = _expected_extracted_dir(extraction_dir, video_path)
    required_files = (
        "kv_store_video_segments.json",
        "kv_store_video_frames.json",
        "kv_store_video_path.json",
    )
    return all(os.path.exists(os.path.join(extracted_dir, name)) for name in required_files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run knowledge_extraction.extractor sequentially for videos in downloads/."
    )
    parser.add_argument(
        "--downloads-dir",
        default=os.path.join(_project_root(), "downloads/queue"),
        help="Directory containing source video files.",
    )
    parser.add_argument(
        "--extraction-dir",
        default=_default_extraction_dir(),
        help="Directory containing extraction outputs (expects extracted_data/<video_basename>/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess videos even if extraction outputs already exist.",
    )
    args = parser.parse_args()

    downloads_dir = os.path.abspath(args.downloads_dir)
    extraction_dir = os.path.abspath(args.extraction_dir)
    if not os.path.isdir(downloads_dir):
        print(f"Downloads directory not found: {downloads_dir}")
        return 1
    if not os.path.isdir(extraction_dir):
        print(f"Extraction directory not found: {extraction_dir}")
        return 1

    videos = _discover_videos(downloads_dir)
    if not videos:
        print(f"No video files found in: {downloads_dir}")
        return 1

    skipped = []
    pending = []
    for video_path in videos:
        if not args.force and _is_video_already_extracted(extraction_dir, video_path):
            skipped.append(video_path)
        else:
            pending.append(video_path)

    print(f"Found {len(videos)} video(s) in {downloads_dir}")
    print(f"Already processed (skipped): {len(skipped)}")
    print(f"Pending: {len(pending)}")

    if not pending:
        print("\nExtraction queue complete.")
        print("No pending videos. All discovered videos are already processed.")
        return 0

    failures = []

    for index, video_path in enumerate(pending, start=1):
        print(f"\n[{index}/{len(pending)}] Processing: {os.path.basename(video_path)}")
        cmd = [
            sys.executable,
            "-m",
            "knowledge_extraction.extractor",
            "--video-path",
            video_path,
        ]
        result = subprocess.run(cmd, cwd=_project_root(), check=False)
        if result.returncode != 0:
            failures.append((video_path, result.returncode))
            print(f"Failed ({result.returncode}): {video_path}")
        else:
            print(f"Completed: {video_path}")

    print("\nExtraction queue complete.")
    if failures:
        print(f"Failures: {len(failures)}")
        for path, code in failures:
            print(f" - [{code}] {path}")
        return 1

    print("All videos processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
