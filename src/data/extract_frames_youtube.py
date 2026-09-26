import argparse
import os
import sys
import cv2
from yt_dlp import YoutubeDL

def download_video(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)
def extract_frames(video_path, out_dir, every_n_seconds=1.0, label="unlabeled"):
    save_dir = os.path.join(out_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(fps * every_n_seconds)))
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    frame_idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(save_dir, f"{video_id}_frame{saved:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1
    cap.release()
    print(f"[{video_id}] saved {saved} frames -> {save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", help="One or more YouTube URLs")
    parser.add_argument("--url_file", help="Text file with one YouTube URL per line")
    parser.add_argument("--out_dir", required=True, help="Root output directory")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between saved frames")
    parser.add_argument("--label", default="unlabeled", help="Subfolder name (e.g. calm/aggressive)")
    parser.add_argument("--keep_video", action="store_true", help="Don't delete downloaded video after extraction")
    args = parser.parse_args()

    urls = list(args.urls) if args.urls else []
    if args.url_file:
        with open(args.url_file) as f:
            urls += [line.strip() for line in f if line.strip()]

    if not urls:
        print("No URLs provided. Use --urls or --url_file.")
        sys.exit(1)

    tmp_video_dir = os.path.join(args.out_dir, "_downloads")

    for url in urls:
        try:
            video_path = download_video(url, tmp_video_dir)
            extract_frames(video_path, args.out_dir, args.interval, args.label)
            if not args.keep_video:
                os.remove(video_path)
        except Exception as e:
            print(f"Failed on {url}: {e}")

if __name__ == "__main__":
    main()