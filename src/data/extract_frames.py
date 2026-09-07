"""
extract_frames_simple.py

Evenly-spaced frame extraction from one or more videos, for manual sorting
into calm/aggressive folders afterward.

Usage:
    python extract_frames_simple.py --input path/to/video_or_folder --output path/to/output_dir --fps 1

    --input   : a single video file OR a folder containing multiple videos
    --output  : folder where extracted frames will be saved
    --fps     : how many frames to extract per second of video (default 1)
                (use a decimal like 0.5 for one frame every 2 seconds)

Frames are saved as: <output>/<video_name>_frame_<index>.jpg

After running, manually review the output folder and sort frames into
train/calm, train/aggressive, val/calm, val/aggressive etc. before running
crop_dataset.py (remember: crop_dataset.py expects that structured layout,
not a flat folder).
"""

import argparse
import os
import cv2
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def extract_from_video(video_path: Path, output_dir: Path, target_fps: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Could not open: {video_path.name}")
        return 0

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # How many source frames to step over to hit the target extraction rate
    frame_interval = max(1, round(source_fps / target_fps))

    video_stem = video_path.stem
    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_name = f"{video_stem}_frame_{saved_count:05d}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"  [OK] {video_path.name}: {saved_count} frames saved "
          f"(source ~{source_fps:.1f} fps, {total_frames} total frames)")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract evenly-spaced frames from video(s).")
    parser.add_argument("--input", required=True, help="Video file or folder of videos")
    parser.add_argument("--output", required=True, help="Output folder for extracted frames")
    parser.add_argument("--fps", type=float, default=1.0,
                         help="Frames to extract per second of video (default: 1.0)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        video_files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )
    else:
        print(f"Input path does not exist: {input_path}")
        return

    if not video_files:
        print("No video files found.")
        return

    print(f"Found {len(video_files)} video(s). Extracting at ~{args.fps} fps each...")
    total_saved = 0
    for video_file in video_files:
        total_saved += extract_from_video(video_file, output_dir, args.fps)

    print(f"\nDone. {total_saved} frames saved to: {output_dir}")
    print("Next step: manually sort these into calm/ and aggressive/ subfolders,")
    print("then arrange into train/val structure before running crop_dataset.py.")


if __name__ == "__main__":
    main()