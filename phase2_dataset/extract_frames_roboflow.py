"""
Extract Frames from Classroom Videos for Roboflow Labeling
===========================================================
Adaptive sampling: 1 fps base + scene-change detection + static-scene dedup.
Target: ~10,000 frames across all videos (~450/video).

Usage:
    python extract_frames_roboflow.py                   # full run
    python extract_frames_roboflow.py --test-run        # first 2 videos only
    python extract_frames_roboflow.py --max-per-video 200  # override limit
"""

import os, sys, argparse, time, json
from pathlib import Path
import cv2
import numpy as np
import contextlib
import io

# ====================== CONFIGURATION ======================

# Source folder with subfolders per day containing .mp4 files
SRC_ROOT = Path(r"F:\Dataset SMP\Training")

# Output root – one subfolder per video
OUT_ROOT = Path(r"F:\Dataset SMP\Training_Frames")

# Sampling
SAMPLE_INTERVAL = 15          # 1 fps at 15-fps video (take every 15th frame)
MAX_FRAMES_PER_VIDEO = 0      # 0 = no limit, extract all from start to end
JPEG_QUALITY = 92             # high quality for Roboflow

# Scene-change detection (SSIM-based)
SSIM_CHANGE_THRESHOLD = 0.85  # capture extra frame if SSIM drops below this
SSIM_DEDUP_THRESHOLD  = 0.98  # skip frame if SSIM is above this (near-duplicate)

# Minimum video file size to process (skip corrupt/tiny files)
MIN_FILE_SIZE_BYTES = 100_000  # 100 KB

# Resize for SSIM computation only (faster), NOT for saved frames
SSIM_RESIZE_DIM = (320, 180)

# Video extensions to look for
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

# ============================================================


def compute_ssim_fast(img_a, img_b):
    """
    Compute a simplified SSIM between two images (resized to SSIM_RESIZE_DIM).
    Uses luminance channel only for speed.
    """
    # Convert to grayscale
    gray_a = cv2.cvtColor(cv2.resize(img_a, SSIM_RESIZE_DIM), cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_b = cv2.cvtColor(cv2.resize(img_b, SSIM_RESIZE_DIM), cv2.COLOR_BGR2GRAY).astype(np.float64)

    C1 = 6.5025    # (0.01 * 255)^2
    C2 = 58.5225   # (0.03 * 255)^2

    mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)

    mu_a_sq  = mu_a * mu_a
    mu_b_sq  = mu_b * mu_b
    mu_a_b   = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(gray_a * gray_a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(gray_b * gray_b, (11, 11), 1.5) - mu_b_sq
    sigma_ab   = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_a_b

    numerator   = (2 * mu_a_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)

    ssim_map = numerator / denominator
    return float(ssim_map.mean())


def collect_videos(src_root):
    """Recursively find all video files, sorted."""
    videos = []
    for p in sorted(Path(src_root).rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            if p.stat().st_size >= MIN_FILE_SIZE_BYTES:
                videos.append(p)
    return videos


def extract_frames_from_video(video_path, out_dir, max_frames, sample_interval):
    """
    Extract frames from a single video using adaptive sampling.

    Returns dict with statistics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    video_name = video_path.stem.replace(" ", "_")

    # Suppress HEVC codec warnings (harmless but noisy)
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": video_name, "status": "FAILED_OPEN", "frames_saved": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Determine actual sample interval based on video fps
    actual_interval = max(int(round(fps / 1.0)), 1)  # target 1 fps
    if sample_interval:
        actual_interval = sample_interval

    saved = 0
    frame_idx = 0
    prev_frame = None
    skipped_dedup = 0
    captured_scene_change = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        should_save = False
        reason = ""

        # Regular interval sampling
        if frame_idx % actual_interval == 0:
            should_save = True
            reason = "interval"

        # Scene change detection (check every 5th frame for performance)
        if not should_save and prev_frame is not None and frame_idx % 5 == 0:
            ssim_val = compute_ssim_fast(frame, prev_frame)
            if ssim_val < SSIM_CHANGE_THRESHOLD:
                should_save = True
                reason = "scene_change"
                captured_scene_change += 1

        if should_save:
            # Dedup check against previous saved frame
            if prev_frame is not None:
                ssim_val = compute_ssim_fast(frame, prev_frame)
                if ssim_val > SSIM_DEDUP_THRESHOLD:
                    skipped_dedup += 1
                    frame_idx += 1
                    continue  # too similar, skip

            # Save frame at original resolution
            out_path = out_dir / f"{video_name}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1
            prev_frame = frame.copy()

            if max_frames > 0 and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()

    return {
        "video": video_name,
        "status": "OK",
        "total_frames_in_video": total_frames,
        "duration_sec": round(duration_sec, 1),
        "fps": round(fps, 1),
        "frames_saved": saved,
        "skipped_dedup": skipped_dedup,
        "captured_scene_change": captured_scene_change,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract frames for Roboflow labeling")
    parser.add_argument("--src", type=str, default=str(SRC_ROOT),
                        help=f"Source video folder (default: {SRC_ROOT})")
    parser.add_argument("--out", type=str, default=str(OUT_ROOT),
                        help=f"Output folder (default: {OUT_ROOT})")
    parser.add_argument("--max-per-video", type=int, default=MAX_FRAMES_PER_VIDEO,
                        help=f"Max frames per video (default: {MAX_FRAMES_PER_VIDEO})")
    parser.add_argument("--sample-interval", type=int, default=SAMPLE_INTERVAL,
                        help=f"Frame sampling interval (default: {SAMPLE_INTERVAL})")
    parser.add_argument("--test-run", action="store_true",
                        help="Process only first 2 videos (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip videos whose output folder already has frames")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FRAME EXTRACTION FOR ROBOFLOW LABELING")
    print("=" * 70)
    print(f"  Source : {src_root}")
    print(f"  Output : {out_root}")
    print(f"  Max/video : {args.max_per_video}")
    print(f"  Interval  : every {args.sample_interval} frames (~{15/args.sample_interval:.1f} fps at 15fps)")
    print(f"  SSIM dedup: > {SSIM_DEDUP_THRESHOLD}")
    print(f"  SSIM scene: < {SSIM_CHANGE_THRESHOLD}")
    print("=" * 70)

    videos = collect_videos(src_root)
    if args.test_run:
        videos = videos[:2]
        print(f"\n[TEST RUN] Processing only first {len(videos)} videos\n")

    print(f"\nFound {len(videos)} videos to process\n")

    all_stats = []
    total_saved = 0
    t_start = time.time()

    for i, vpath in enumerate(videos, 1):
        video_name = vpath.stem.replace(" ", "_")
        video_out = out_root / video_name

        # Resume support
        if args.resume and video_out.exists():
            existing = len(list(video_out.glob("*.jpg")))
            if existing > 0:
                print(f"[{i}/{len(videos)}] SKIP (resume) {vpath.name} — {existing} frames already exist")
                all_stats.append({
                    "video": video_name, "status": "SKIPPED_RESUME",
                    "frames_saved": existing
                })
                total_saved += existing
                continue

        print(f"[{i}/{len(videos)}] Processing: {vpath.name} ...", end=" ", flush=True)
        t0 = time.time()

        stats = extract_frames_from_video(
            video_path=vpath,
            out_dir=video_out,
            max_frames=args.max_per_video,
            sample_interval=args.sample_interval,
        )

        elapsed = time.time() - t0
        print(f"✓ {stats['frames_saved']} frames saved "
              f"(dedup skipped: {stats.get('skipped_dedup', 0)}, "
              f"scene changes: {stats.get('captured_scene_change', 0)}) "
              f"[{elapsed:.1f}s]")

        all_stats.append(stats)
        total_saved += stats["frames_saved"]

    total_time = time.time() - t_start

    # Write report
    report_path = out_root / "extraction_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FRAME EXTRACTION REPORT\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {src_root}\n")
        f.write(f"Output: {out_root}\n")
        f.write(f"Total videos: {len(videos)}\n")
        f.write(f"Total frames saved: {total_saved}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Settings: interval={args.sample_interval}, max/video={args.max_per_video}\n")
        f.write(f"SSIM dedup={SSIM_DEDUP_THRESHOLD}, scene_change={SSIM_CHANGE_THRESHOLD}\n\n")
        f.write("-" * 60 + "\n")
        for s in all_stats:
            f.write(f"{s['video']}: {s['frames_saved']} frames ({s['status']})\n")

    # Also save as JSON for programmatic use
    json_path = out_root / "extraction_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "settings": {
                "sample_interval": args.sample_interval,
                "max_per_video": args.max_per_video,
                "ssim_dedup": SSIM_DEDUP_THRESHOLD,
                "ssim_change": SSIM_CHANGE_THRESHOLD,
                "jpeg_quality": JPEG_QUALITY,
            },
            "total_videos": len(videos),
            "total_frames": total_saved,
            "total_time_sec": round(total_time, 1),
            "videos": all_stats,
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("  EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Videos processed : {len(videos)}")
    print(f"  Total frames     : {total_saved:,}")
    print(f"  Time elapsed     : {total_time:.1f}s")
    print(f"  Avg frames/video : {total_saved / max(len(videos), 1):.0f}")
    print(f"  Report           : {report_path}")
    print(f"  Output           : {out_root}")
    print("=" * 70)

    if total_saved < 8000:
        print(f"\n[TIP] Got {total_saved} frames. To get closer to 10k, try:")
        print(f"  --sample-interval 15  (higher sample rate)")
        print(f"  --max-per-video 600   (raise per-video cap)")


if __name__ == "__main__":
    main()
