"""
Inventarisasi sesi di crops_v7 (train + valid + test).

Tujuan: lihat berapa sesi unik yang tersedia, distribusi tiap sesi per kelas,
dan distribusi original vs augmented (dari nama file). Output ini jadi dasar
untuk merancang split train/val/test multi-session di V10.

Jalankan: python phase3_training/inventory_sessions_v7.py
"""

from pathlib import Path
from collections import defaultdict
import re

ROOT = Path("phase2_dataset/crops_v7")
SPLITS = ["train", "valid", "test"]
CLASSES = ["Engaged", "NotEngaged"]
SESSION_PAT = re.compile(r"(Kelas\d+_\d+[a-zA-Z]+_\d+)", re.IGNORECASE)
AUG_PAT = re.compile(r"_aug|aug_", re.IGNORECASE)


def parse(filename: str):
    name = Path(filename).stem
    m = SESSION_PAT.search(name)
    session = m.group(1) if m else "UNKNOWN_" + name[:30]
    is_aug = bool(AUG_PAT.search(name))
    return session, is_aug


# session -> {(split, cls): {"orig": n, "aug": n}}
table = defaultdict(lambda: defaultdict(lambda: {"orig": 0, "aug": 0}))

for split in SPLITS:
    for cls in CLASSES:
        d = ROOT / split / cls
        if not d.exists():
            continue
        for img in d.glob("*.*"):
            session, is_aug = parse(img.name)
            key = "aug" if is_aug else "orig"
            table[session][(split, cls)][key] += 1

# Print summary
all_sessions = sorted(table.keys())
print(f"Total sesi unik di crops_v7: {len(all_sessions)}\n")

header = f"{'Session':<32} | "
for split in SPLITS:
    for cls in CLASSES:
        header += f"{split[:3]}.{cls[:3]:<4}"
        header += "(o/a) "
print(header)
print("-" * len(header))

totals = defaultdict(lambda: {"orig": 0, "aug": 0})
for sess in all_sessions:
    row = f"{sess:<32} | "
    for split in SPLITS:
        for cls in CLASSES:
            v = table[sess][(split, cls)]
            row += f"{v['orig']:>4}/{v['aug']:<4} "
            totals[(split, cls)]["orig"] += v["orig"]
            totals[(split, cls)]["aug"] += v["aug"]
    print(row)

print("-" * len(header))
trow = f"{'TOTAL':<32} | "
for split in SPLITS:
    for cls in CLASSES:
        v = totals[(split, cls)]
        trow += f"{v['orig']:>4}/{v['aug']:<4} "
print(trow)

# Per-session totals (gabungan semua split, original only)
print("\n=== Per-sesi: jumlah ORIGINAL crops (gabungan train+val+test) ===")
print(f"{'Session':<32} | {'Engaged':>8} {'NotEngaged':>11} {'Total':>7}")
print("-" * 65)
sess_totals = []
for sess in all_sessions:
    eng = sum(table[sess][(s, "Engaged")]["orig"] for s in SPLITS)
    noteng = sum(table[sess][(s, "NotEngaged")]["orig"] for s in SPLITS)
    sess_totals.append((sess, eng, noteng, eng + noteng))

for sess, eng, noteng, tot in sorted(sess_totals, key=lambda x: -x[3]):
    print(f"{sess:<32} | {eng:>8} {noteng:>11} {tot:>7}")

print(f"\nGRAND TOTAL ORIGINAL: Engaged={sum(s[1] for s in sess_totals)}, NotEngaged={sum(s[2] for s in sess_totals)}")
