import os, random, subprocess, sys
from pathlib import Path
import numpy as np
import cv2

# --- Konfigurasi Utama ---
SRC = {
    "High":  [r"F:\OUC-CGE dataset\high"],
    "Medium":[r"F:\OUC-CGE dataset\med"],
    "Low":   [r"F:\OUC-CGE dataset\Low"],
}
OUT_ROOT = Path(r"D:\kuliah\Skripsi\training dataset\ouc-cge-cls")

# Sampling & ukuran
TARGET_FPS = 4                 # 4 fps (naikkan/kurangi sesuai kebutuhan)
FRAMES_CAP_PER_VIDEO = 12      # BATASI 12 frame/video biar hemat disk (bisa dinaikkan nanti)
JPEG_QUALITY = 88              # 85–92 cukup, 88 hemat ukuran
SIZE_MIN_BYTES = 50_000        # skip video < ~50KB (kemungkinan korup)

# Split per-VIDEO (hindari leakage)
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Ekstensi video yang dikenali
VIDEO_EXTS = {".mp4",".avi",".mov",".mkv",".wmv",".flv",".m4v",".3gp"}

# OpenCV: naikkan batas attempt (sebelum load video)
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "20000")

# Log bad videos
LOG_BAD = OUT_ROOT / "bad_videos.txt"
SKIP_IF_EXISTS = True  # resume-able: jika frame tujuan sudah ada, skip video tsb

def collect_videos(roots):
    vids = []
    for r in roots:
        r = Path(r)
        if not r.exists():
            print(f"[WARN] missing: {r}")
            continue
        for p in r.rglob("*"):
            if p.suffix.lower() in VIDEO_EXTS and p.is_file():
                # filter ukuran minimal
                try:
                    if p.stat().st_size < SIZE_MIN_BYTES:
                        continue
                except Exception:
                    pass
                vids.append(p)
    return vids

def ensure_dirs():
    for split in SPLITS:
        for cls in SRC:
            (OUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

def write_jpg(path: Path, frame_bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

def base_name(video_path: Path) -> str:
    return video_path.stem.replace(" ", "_")

def already_extracted(dest_dir: Path, base: str) -> bool:
    if not SKIP_IF_EXISTS:
        return False
    # Jika sudah ada minimal 1 frame untuk video ini, anggap selesai (resume)
    return any(dest_dir.glob(f"{base}_*.*"))

# --- Layer 1: OpenCV ---
def extract_with_opencv(video_path: Path, dest_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        fps = 30.0
    step = max(int(round(fps / TARGET_FPS)), 1)
    saved = 0
    idx = 0
    b = base_name(video_path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out = dest_dir / f"{b}_cv2_{idx:06d}.jpg"
            write_jpg(out, frame)
            saved += 1
            if saved >= FRAMES_CAP_PER_VIDEO:
                break
        idx += 1
    cap.release()
    return saved

# --- Layer 2: imageio-ffmpeg ---
def extract_with_imageio(video_path: Path, dest_dir: Path) -> int:
    try:
        import imageio.v3 as iio
    except Exception:
        return -1
    try:
        meta = iio.immeta(video_path, plugin="ffmpeg")
        fps = float(meta.get("fps", 30.0)) or 30.0
        step = max(int(round(fps / TARGET_FPS)), 1)
        saved = 0
        b = base_name(video_path)
        for i, frame in enumerate(iio.imiter(video_path, plugin="ffmpeg")):
            if i % step != 0:
                continue
            # imageio -> RGB
            frame_bgr = frame[..., ::-1]
            out = dest_dir / f"{b}_iio_{i:06d}.jpg"
            write_jpg(out, frame_bgr)
            saved += 1
            if saved >= FRAMES_CAP_PER_VIDEO:
                break
        return saved
    except Exception:
        return -1

# --- Layer 3: Decord ---
def extract_with_decord(video_path: Path, dest_dir: Path) -> int:
    try:
        from decord import VideoReader, cpu
    except Exception:
        return -1
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        T = len(vr)
        if T <= 0:
            return -1
        take = min(FRAMES_CAP_PER_VIDEO, T)
        idxs = np.linspace(0, T-1, num=take, dtype=int)
        b = base_name(video_path)
        saved = 0
        for idx in idxs:
            frame_rgb = vr[idx].asnumpy()
            frame_bgr = frame_rgb[..., ::-1]
            out = dest_dir / f"{b}_decord_{idx:06d}.jpg"
            write_jpg(out, frame_bgr)
            saved += 1
        return saved
    except Exception:
        return -1

# --- Layer 4: FFmpeg CLI direct (via imageio-ffmpeg binary) ---
def extract_with_ffmpeg_cli(video_path: Path, dest_dir: Path) -> int:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except Exception:
        return -1
    ffmpeg = get_ffmpeg_exe()
    b = base_name(video_path)
    out_pat = str(dest_dir / f"{b}_ff_%06d.jpg")
    # Langsung ekstrak n frame pada fps target
    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-err_detect", "ignore_err",
        "-probesize", "200M", "-analyzeduration", "200M",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-vf", f"fps={TARGET_FPS}",
        "-frames:v", str(FRAMES_CAP_PER_VIDEO),
        "-q:v", str(31 - (JPEG_QUALITY // 3)),  # approx: kualitas -> qscale
        "-y", out_pat
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return -1
        # hitung output yang jadi
        count = len(list(dest_dir.glob(f"{b}_ff_*.jpg")))
        return count
    except Exception:
        return -1

def extract_frames(video_path: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    b = base_name(video_path)
    if already_extracted(dest_dir, b):
        return 1  # anggap sudah beres (resume)
    # Try 4 lapis
    for fn in (extract_with_opencv, extract_with_imageio, extract_with_decord, extract_with_ffmpeg_cli):
        n = fn(video_path, dest_dir)
        if n and n > 0:
            return n
    return -1

def main():
    ensure_dirs()
    bad = []
    total_saved = 0

    # Kumpulkan video per kelas
    all_videos = {}
    for cls, roots in SRC.items():
        vids = collect_videos(roots)
        random.shuffle(vids)
        all_videos[cls] = vids
        print(f"{cls}: {len(vids)} videos")

    # Split per-VIDEO → 80/10/10
    for cls, vids in all_videos.items():
        n = len(vids)
        n_train = int(SPLITS["train"] * n)
        n_val   = int(SPLITS["val"]   * n)
        splits = {
            "train": vids[:n_train],
            "val":   vids[n_train:n_train+n_val],
            "test":  vids[n_train+n_val:]
        }

        cls_saved = 0
        processed = 0
        for split, vlist in splits.items():
            dest = OUT_ROOT / split / cls
            for v in vlist:
                processed += 1
                try:
                    saved = extract_frames(v, dest)
                except Exception:
                    saved = -1
                if saved <= 0:
                    bad.append(str(v))
                else:
                    cls_saved += saved
                    total_saved += saved
                if processed % 100 == 0:
                    print(f"[{cls}] processed={processed}/{n}  frames_saved={cls_saved}")

        print(f"[{cls}] frames_saved={cls_saved}, bad_videos={len([x for x in bad if x])}")

    if bad:
        LOG_BAD.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_BAD, "w", encoding="utf-8") as f:
            f.write("\n".join(bad))
        print(f"\n[INFO] {len(bad)} bad videos logged to: {LOG_BAD}")

    print(f"\n✔ DONE. Total frames saved: {total_saved}")
    print("Output:", OUT_ROOT)

if __name__ == "__main__":
    main()