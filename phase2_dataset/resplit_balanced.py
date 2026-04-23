"""
Buat dataset_smp_balanced dari dataset_smp.

Perubahan: Kelas9_2mar_1012 dipindah dari train → valid
sehingga val set menjadi ~50:50 Engaged:NotEngaged.

  Val sebelum : 1805 Engaged | 1019 NotEngaged (64:36)
  Val sesudah : 1805 Engaged | 1813 NotEngaged (50:50) ✅

Tidak ada session bleeding — sesi dipindah utuh.
dataset_smp asli tidak disentuh.
"""

import shutil
import collections
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
SRC     = ROOT / "dataset_smp"
DST     = ROOT / "dataset_smp_balanced"
SPLITS  = ["train", "valid", "test"]

MOVE_TO_VALID = {"Kelas9_2mar_1012"}   # session key yang dipindah train → valid


def session_of(stem: str) -> str:
    """Ekstrak session key dari nama file (sebelum _frame_)."""
    idx = stem.find("_frame_")
    return stem[:idx] if idx != -1 else stem


def build():
    if DST.exists():
        print(f"[WARN] {DST.name} sudah ada. Hapus dulu jika ingin rebuild:\n  rm -rf {DST}")
        return

    print(f"Source : {SRC.name}")
    print(f"Output : {DST.name}")
    print(f"Sesi dipindah ke valid: {MOVE_TO_VALID}\n")

    counts = collections.defaultdict(lambda: collections.Counter())
    moved  = collections.Counter()

    for split in SPLITS:
        src_img = SRC  / split / "images"
        src_lbl = SRC  / split / "labels"

        for img_path in sorted(src_img.glob("*")):
            sess = session_of(img_path.stem)

            # Tentukan split tujuan
            dst_split = split
            if split == "train" and sess in MOVE_TO_VALID:
                dst_split = "valid"
                moved[sess] += 1

            # Salin gambar
            dst_img = DST / dst_split / "images"
            dst_img.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img / img_path.name)

            # Salin label (jika ada)
            lbl_path = src_lbl / (img_path.stem + ".txt")
            if lbl_path.exists():
                dst_lbl = DST / dst_split / "labels"
                dst_lbl.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lbl_path, dst_lbl / lbl_path.name)

            counts[dst_split]["images"] += 1

    # Salin data.yaml dengan path yang diupdate
    yaml_src = SRC / "data.yaml"
    if yaml_src.exists():
        shutil.copy2(yaml_src, DST / "data.yaml")

    # Laporan
    print("=== HASIL ===")
    CLASS_MAP = {0: "High", 1: "Low", 2: "Medium"}
    for split in SPLITS:
        lbl_dir = DST / split / "labels"
        cls_counts = collections.Counter()
        for lbl in lbl_dir.glob("*.txt"):
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = CLASS_MAP.get(int(parts[0]))
                    if cls:
                        cls_counts[cls] += 1
        total = sum(cls_counts.values())
        engaged    = cls_counts["High"] + cls_counts["Medium"]
        notengaged = cls_counts["Low"]
        tot2 = engaged + notengaged
        print(f"\n  [{split.upper()}]  {counts[split]['images']} gambar | {total} anotasi")
        print(f"    Engaged    (High+Med): {engaged:5d}  ({100*engaged/tot2:.1f}%)")
        print(f"    NotEngaged (Low)     : {notengaged:5d}  ({100*notengaged/tot2:.1f}%)")
        print(f"    High: {cls_counts['High']}  Low: {cls_counts['Low']}  Medium: {cls_counts['Medium']}")

    print(f"\nSesi yang dipindah ke valid:")
    for sess, n in moved.items():
        print(f"  {sess}: {n} gambar")

    print(f"\nSelesai → {DST}")


if __name__ == "__main__":
    build()
