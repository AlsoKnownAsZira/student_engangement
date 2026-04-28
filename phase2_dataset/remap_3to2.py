"""
Remap dataset 3-class (High=0, Low=1, Medium=2) ke 2-class untuk V5:
  - Engaged     = 0  (dari High=0 dan Medium=2)
  - NotEngaged  = 1  (dari Low=1)

Input  : dataset_yolo_split/{train,valid,test}/{images,labels}
Output : dataset_yolo_v5/{train,valid,test}/{images,labels}  + data.yaml

Images di-hardlink (fallback copy) — hemat disk.
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "dataset_yolo_split"
DST  = ROOT / "dataset_yolo_v5"

# Old id -> New id
REMAP = {
    0: 0,  # High    -> Engaged
    2: 0,  # Medium  -> Engaged
    1: 1,  # Low     -> NotEngaged
}

SPLITS = ["train", "valid", "test"]


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remap_label_file(src_txt: Path, dst_txt: Path):
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    lines_out = []
    for line in src_txt.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        old_cls = int(parts[0])
        if old_cls not in REMAP:
            continue
        parts[0] = str(REMAP[old_cls])
        lines_out.append(" ".join(parts))
    dst_txt.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))


def process_split(split: str):
    src_img_dir = SRC / split / "images"
    src_lbl_dir = SRC / split / "labels"
    dst_img_dir = DST / split / "images"
    dst_lbl_dir = DST / split / "labels"

    counts = {0: 0, 1: 0}
    n_images = 0

    for lbl in src_lbl_dir.glob("*.txt"):
        stem = lbl.stem
        # Match image (jpg/jpeg/png)
        img_src = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = src_img_dir / f"{stem}{ext}"
            if cand.exists():
                img_src = cand
                break
        if img_src is None:
            print(f"  [WARN] gambar tidak ditemukan untuk {stem}")
            continue

        link_or_copy(img_src, dst_img_dir / img_src.name)
        remap_label_file(lbl, dst_lbl_dir / lbl.name)
        n_images += 1

        for line in (dst_lbl_dir / lbl.name).read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1

    total = sum(counts.values())
    print(f"\n[{split.upper()}] {n_images} images, {total} annotations")
    if total:
        for cid, name in [(0, "Engaged"), (1, "NotEngaged")]:
            pct = counts[cid] / total * 100
            print(f"  {name:11s}: {counts[cid]:5d} ({pct:5.1f}%)")


def write_yaml():
    yaml_path = DST / "data.yaml"
    yaml_path.write_text(
        f"path: {DST.as_posix()}\n"
        "train: train/images\n"
        "val:   valid/images\n"
        "test:  test/images\n"
        "\n"
        "nc: 2\n"
        "names: ['Engaged', 'NotEngaged']\n"
    )
    print(f"\n[OK] data.yaml ditulis: {yaml_path}")


if __name__ == "__main__":
    assert SRC.exists(), f"Source dataset tidak ditemukan: {SRC}"
    print(f"Source : {SRC}")
    print(f"Target : {DST}")

    for split in SPLITS:
        process_split(split)

    write_yaml()
    print("\nSelesai. Zip folder dataset_yolo_v5/ dan upload ke Kaggle.")
