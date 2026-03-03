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


def _discover_videos(downloads_dir: str) -> List[str]:
    videos = []
    for entry in os.scandir(downloads_dir):
        if not entry.is_file():
            continue
        ext = os.path.splitext(entry.name)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            videos.append(entry.path)
    return sorted(videos)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run knowledge_extraction.extractor sequentially for videos in downloads/."
    )
    parser.add_argument(
        "--downloads-dir",
        default=os.path.join(_project_root(), "downloads/queue"),
        help="Directory containing source video files.",
    )
    args = parser.parse_args()

    downloads_dir = os.path.abspath(args.downloads_dir)
    if not os.path.isdir(downloads_dir):
        print(f"Downloads directory not found: {downloads_dir}")
        return 1

    videos = _discover_videos(downloads_dir)
    if not videos:
        print(f"No video files found in: {downloads_dir}")
        return 1

    print(f"Found {len(videos)} video(s) in {downloads_dir}")
    failures = []

    for index, video_path in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}] Processing: {os.path.basename(video_path)}")
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
