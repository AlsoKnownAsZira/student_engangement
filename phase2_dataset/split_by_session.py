"""
split_by_session.py
====================
Membagi dataset hasil export Roboflow (format YOLO v11) berdasarkan sesi rekaman
yang dikodekan dalam nama file → mencegah data leakage antar split.

Aturan:
  - Semua frame diekspor dari Roboflow sebagai 100% Train
  - Script ini me-reorganisasi folder output menjadi:
      output_dir/
        train/  images/ + labels/
        valid/  images/ + labels/
        test/   images/ + labels/
  - Split dilakukan per SESSION UNIK, bukan per frame secara acak
  - Sesi diidentifikasi dari pattern: Kelas<X>_<Tanggal>_<Jam>
    Contoh filename: Kelas9_6mar_0959_frame_000150.jpg
                     ^^^^^^^^^^^^^^^^
                     → session_id = "Kelas9_6mar_0959"

Proporsi split default: 70% Train / 20% Valid / 10% Test (per jumlah sesi)

Usage:
    python split_by_session.py \\
        --input  <folder_export_roboflow> \\
        --output <folder_output_split>    \\
        [--train 10] [--valid 3] [--test 1] \\
        [--seed 42]

Catatan:
    --train / --valid / --test adalah JUMLAH SESI (bukan persentase).
    Jika tidak diisi, script akan menghitung otomatis dari proporsi 70/20/10.
"""

import argparse
import os
import re
import shutil
import random
from collections import defaultdict
from pathlib import Path


# ─── Regex ──────────────────────────────────────────────────────────────────
# Cocok dengan pola: Kelas<angka>_<tanggal>_<jam>
# Misal: Kelas9_6mar_0959 atau Kelas10_15feb_1330
SESSION_PATTERN = re.compile(
    r"(Kelas\d+_\d+\w+_\d{4})(?:_frame|$)",
    re.IGNORECASE,
)


def extract_session_id(filename: str) -> str | None:
    """Ekstrak session_id dari nama file.

    Contoh:
        'Kelas9_6mar_0959_frame_000150.jpg' → 'Kelas9_6mar_0959'
    Kembalikan None jika pattern tidak ditemukan.
    """
    stem = Path(filename).stem
    match = SESSION_PATTERN.search(stem)
    return match.group(1) if match else None


def collect_sessions(images_dir: Path) -> dict[str, list[Path]]:
    """Kumpulkan semua image paths dan kelompokkan berdasarkan session_id."""
    sessions: dict[str, list[Path]] = defaultdict(list)
    unmatched: list[Path] = []

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        sid = extract_session_id(img_path.name)
        if sid:
            sessions[sid].append(img_path)
        else:
            unmatched.append(img_path)

    if unmatched:
        print(f"\n[WARN] {len(unmatched)} file tidak cocok dengan pattern session:")
        for p in unmatched[:10]:
            print(f"       {p.name}")
        if len(unmatched) > 10:
            print(f"       ... dan {len(unmatched) - 10} lagi")

    return dict(sessions)


def compute_split_counts(
    n_sessions: int,
    n_train: int | None,
    n_valid: int | None,
    n_test: int | None,
) -> tuple[int, int, int]:
    """Hitung jumlah sesi untuk masing-masing split."""
    if n_train is not None and n_valid is not None and n_test is not None:
        total = n_train + n_valid + n_test
        if total > n_sessions:
            raise ValueError(
                f"Jumlah sesi yang diminta ({total}) > sesi tersedia ({n_sessions})"
            )
        return n_train, n_valid, n_test

    # Hitung otomatis dengan proporsi 70 / 20 / 10
    n_test_calc  = max(1, round(n_sessions * 0.10))
    n_valid_calc = max(1, round(n_sessions * 0.20))
    n_train_calc = n_sessions - n_valid_calc - n_test_calc
    if n_train_calc < 1:
        raise ValueError(
            f"Tidak cukup sesi ({n_sessions}) untuk dibagi menjadi 3 split."
        )
    return n_train_calc, n_valid_calc, n_test_calc


def copy_files(
    image_paths: list[Path],
    labels_dir: Path,
    dest_images_dir: Path,
    dest_labels_dir: Path,
    missing_label_ok: bool = True,
) -> tuple[int, int]:
    """Copy image + label ke folder tujuan. Kembalikan (copied, skipped)."""
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            if missing_label_ok:
                skipped += 1
                continue
            else:
                raise FileNotFoundError(f"Label tidak ditemukan: {label_path}")

        shutil.copy2(img_path, dest_images_dir / img_path.name)
        shutil.copy2(label_path, dest_labels_dir / label_path.name)
        copied += 1

    return copied, skipped


def write_data_yaml(
    output_dir: Path,
    class_names: list[str] | None = None,
) -> None:
    """Tulis data.yaml untuk YOLO training."""
    # Default class names sesuai proyek
    if class_names is None:
        class_names = ["High", "Low", "Medium"]

    yaml_content = f"""\
path: {output_dir.resolve().as_posix()}
train: train/images
val:   valid/images
test:  test/images

nc: {len(class_names)}
names: {class_names}
"""
    (output_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\n[INFO] data.yaml ditulis ke {output_dir / 'data.yaml'}")


def print_summary(
    sessions: dict[str, list[Path]],
    train_sessions: list[str],
    valid_sessions: list[str],
    test_sessions: list[str],
    counts: dict[str, tuple[int, int]],
) -> None:
    total_frames = sum(len(v) for v in sessions.values())

    print("\n" + "=" * 60)
    print("  RINGKASAN SPLIT BY SESSION")
    print("=" * 60)
    print(f"  Total sesi unik  : {len(sessions)}")
    print(f"  Total frame      : {total_frames}")
    print()

    for split_name, split_sessions in [
        ("TRAIN", train_sessions),
        ("VALID", valid_sessions),
        ("TEST",  test_sessions),
    ]:
        copied, skipped = counts[split_name.lower()]
        frame_total = sum(len(sessions[s]) for s in split_sessions)
        print(f"  {split_name} ({len(split_sessions)} sesi, {frame_total} frame):")
        for s in split_sessions:
            print(f"    · {s}  ({len(sessions[s])} frame)")
        print(f"    → {copied} file di-copy, {skipped} dilewati (tidak ada label)")
        print()

    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split dataset YOLO berdasarkan sesi rekaman (anti data leakage)"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Folder hasil export Roboflow (berisi train/images & train/labels)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Folder output hasil split (train/valid/test)"
    )
    parser.add_argument(
        "--train", type=int, default=None,
        help="Jumlah sesi untuk Train (default: auto 70%%)"
    )
    parser.add_argument(
        "--valid", type=int, default=None,
        help="Jumlah sesi untuk Valid (default: auto 20%%)"
    )
    parser.add_argument(
        "--test", type=int, default=None,
        help="Jumlah sesi untuk Test (default: auto 10%%)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed untuk reprodusibilitas (default: 42)"
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Nama class (default: High Low Medium)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Tampilkan rencana split tanpa menyalin file"
    )
    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    # Roboflow export biasanya meletakkan semua frame di train/images + train/labels
    # (karena kita set 100% train di Roboflow)
    images_dir = input_dir / "train" / "images"
    labels_dir = input_dir / "train" / "labels"

    # Fallback: cek apakah image berada langsung di input/images
    if not images_dir.exists():
        images_dir = input_dir / "images"
        labels_dir = input_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Folder images tidak ditemukan. Dicoba:\n"
            f"  {input_dir / 'train' / 'images'}\n"
            f"  {input_dir / 'images'}"
        )

    print(f"[INFO] Input images : {images_dir}")
    print(f"[INFO] Input labels : {labels_dir}")
    print(f"[INFO] Output dir   : {output_dir}")

    # ── Kumpulkan & kelompokkan per sesi ─────────────────────────────────────
    sessions = collect_sessions(images_dir)
    if not sessions:
        print("[ERROR] Tidak ada file yang cocok dengan pattern session. Periksa nama file.")
        return

    all_session_ids = sorted(sessions.keys())
    print(f"\n[INFO] Sesi unik ditemukan ({len(all_session_ids)}):")
    for sid in all_session_ids:
        print(f"       {sid}  ({len(sessions[sid])} frame)")

    # ── Hitung distribusi split ────────────────────────────────────────────
    n_train, n_valid, n_test = compute_split_counts(
        len(all_session_ids), args.train, args.valid, args.test
    )

    # ── Random shuffle → assign ke split ──────────────────────────────────
    random.seed(args.seed)
    shuffled = all_session_ids.copy()
    random.shuffle(shuffled)

    test_sessions  = shuffled[:n_test]
    valid_sessions = shuffled[n_test : n_test + n_valid]
    train_sessions = shuffled[n_test + n_valid :]

    if args.dry_run:
        print("\n[DRY RUN] Tidak ada file yang disalin. Rencana split:")
        for name, group in [
            ("TRAIN", train_sessions),
            ("VALID", valid_sessions),
            ("TEST",  test_sessions),
        ]:
            frames = sum(len(sessions[s]) for s in group)
            print(f"  {name}: {len(group)} sesi, {frames} frame")
            for s in group:
                print(f"    · {s}")
        return

    # ── Copy files ────────────────────────────────────────────────────────
    counts: dict[str, tuple[int, int]] = {}
    for split_name, split_sessions in [
        ("train", train_sessions),
        ("valid", valid_sessions),
        ("test",  test_sessions),
    ]:
        all_paths = []
        for sid in split_sessions:
            all_paths.extend(sessions[sid])

        dest_imgs   = output_dir / split_name / "images"
        dest_labels = output_dir / split_name / "labels"

        copied, skipped = copy_files(all_paths, labels_dir, dest_imgs, dest_labels)
        counts[split_name] = (copied, skipped)
        print(f"[INFO] {split_name.upper()}: {copied} file di-copy ({skipped} diabaikan)")

    # ── data.yaml ─────────────────────────────────────────────────────────
    write_data_yaml(output_dir, class_names=args.classes)

    # ── Ringkasan ─────────────────────────────────────────────────────────
    print_summary(sessions, train_sessions, valid_sessions, test_sessions, counts)
    print(f"\n[DONE] Dataset siap di: {output_dir}")


if __name__ == "__main__":
    main()
