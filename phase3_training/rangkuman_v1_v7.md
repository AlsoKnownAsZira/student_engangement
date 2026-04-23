# Rangkuman Percobaan Training V1–V9

## Overview

| Versi | Task | Model | Resolusi | Dataset | Metrik Utama (Test) | Catatan Utama |
|-------|------|-------|----------|---------|---------------------|---------------|
| V1 | Detect (3-class) | YOLOv11s | 640px | Original (742 img) | mAP@50 ~57.8% | Baseline |
| V2 | Detect (3-class) | YOLOv11m | 960px | Original (742 img) | mAP@50 58.5% | +SAHI (dibuang) |
| V3 | Detect (3-class) | YOLOv11m | 1280px | Original (742 img) | mAP@50 ~58% | Resolusi tinggi, tidak membantu |
| V4 | Detect (3-class) | YOLOv11m | 960px | Balanced V4 (1818 img) | mAP@50 54.8% | Augment frame offline, turun |
| V5 | Detect (2-class) | YOLOv11m | 960px | Remap 742 img → 2cls | mAP@50 76.2% | Merge High+Med=Engaged, breakthrough |
| V6 | Classify (3-class) | YOLOv11s-cls | 224px | crops_v6 (30k train) | Top-1 Acc 57.7% | Crop per orang, balanced 10k/kelas |
| V7 | **Classify (2-class)** | YOLOv11s-cls | 224px | crops_v7 (16.7k train) | Top-1 Acc 76.8% | Crop + 2-class, Low di-augment offline |
| V8 | Classify (2-class) | YOLOv11s-cls | 224px | crops_v8 (16.8k train) | Top-1 Acc **73.0%** | Dataset baru (Low 6130), best epoch 6, turun dari V7 |
| V9 | Classify (2-class) | YOLOv11s-cls | 320px | crops_v9 (val balanced 50:50) | Top-1 Acc **70.6%** | AdamW+imgsz=320, val acc naik (0.836) tapi test turun, NotEngaged bias |

---

## Per-Class Metrics — Test Set

> V1–V5: mAP@50 (detection). V6–V7: F1-score (classification). Tidak apple-to-apple tapi menunjukkan tren per kelas.

| Kelas | V1 | V2 | V3 | V4 | V5 | V6 | V7 | V8 | V9 |
|-------|----|----|----|----|----|----|-----|----|----|
| High / Engaged    | ~79.4% | 72.0% | ~72% | 66.9% | mAP **85.3%** | F1 0.569 | F1 **0.802** | F1 0.752 | F1 0.723 |
| Low / NotEngaged  | ~57.2% | 62.1% | ~60% | 58.5% | mAP 67.0% | F1 0.698 | F1 0.721 | F1 0.703 | F1 0.687 |
| Medium            | ~36.6% | 41.4% | ~38% | 38.9% | *(merged)* | F1 0.335 | *(merged)* | *(merged)* | *(merged)* |
| **Overall**       | **~57.8%** | **58.5%** | **~58%** | **54.8%** | **mAP 76.2%** | **Acc 57.7%** | **Acc 76.8%** | **Acc 73.0%** | **Acc 70.6%** |

---

## Detail Per Versi

### V1 — Baseline
- **Model:** YOLOv11s (9.4M params)
- **Resolusi:** 640px, Batch: 16, Patience: 15, Epochs: 70
- **Dataset:** Original 742 img, 3-class (High/Low/Medium)
- **Masalah:** Model terlalu kecil, resolusi terlalu rendah. 63% bounding box siswa <1% area frame. Patience=15 terlalu agresif.
- **Hasil:** mAP ~57.8%, Medium 36.6%

---

### V2 — Upgrade Model + Resolusi
- **Model:** YOLOv11m (20M params)
- **Resolusi:** 960px, Batch: 8, Patience: 30, cls=1.0, dropout=0.1, copy_paste=0.1
- **Dataset:** Original 742 img, 3-class
- **Hipotesis:** Model lebih besar + resolusi lebih tinggi akan menembus plateau V1.
- **Hasil:** mAP 58.5% — naik tipis 0.7%, tidak signifikan.
- **Tambahan:** SAHI dicoba untuk deteksi siswa kecil → **dibuang** karena tile overlap → ID explosion 58→131.
- **Kesimpulan:** Plateau bukan karena model atau resolusi. Data training tidak berubah = hasil tidak berubah.

---

### V3 — Resolusi Lebih Tinggi
- **Model:** YOLOv11m
- **Resolusi:** 1280px, Batch: 4, Patience: 40, cls=1.5
- **Dataset:** Original 742 img, 3-class
- **Hipotesis:** 1280px akan bantu deteksi siswa kecil lebih baik dari 960px.
- **Hasil:** ~58% — identik dengan V2, lebih lambat.
- **Kesimpulan:** 1280px tidak membantu. Bottleneck bukan resolusi tapi distribusi data.

---

### V4 — Balanced Dataset (Augment Offline Frame-Level)
- **Model:** YOLOv11m
- **Resolusi:** 960px, Batch: 6, Patience: 70, cls=1.5, copy_paste=0.1, label_smoothing=0.1, close_mosaic=30
- **Dataset:** train_v4 — augmentasi offline targeted Low dan Medium
  - High: 12274 (37.5%), Low: 10544 (32.2%), Medium: 9929 (30.3%)
  - Total 1818 images, 32747 annotations
- **Catatan:** Sesi Colab disconnected di epoch 69. best.pt epoch 22 (val mAP 0.639) berhasil disimpan.
- **Hasil:** mAP 54.8% (standard), 56.7% (TTA) — **turun dari V1-V3**
- **Kesimpulan:** Augment di level frame tidak efektif — satu frame berisi campuran semua kelas, augment frame = semua kelas ikut naik bersama. Val spike 0.639 adalah noise dari val set kecil.

---

### V5 — 2-Class Detection (Engaged / NotEngaged)
- **Model:** YOLOv11m
- **Resolusi:** 960px, Batch: 8, Patience: 50, cls=1.2, label_smoothing=0.05, copy_paste=0.2
- **Dataset:** dataset_yolo_v5 — remap dari original 742 img
  - High + Medium → Engaged (68.3%), Low → NotEngaged (31.7%)
- **Hipotesis:** Ambiguitas kelas Medium adalah bottleneck utama — merge dengan High untuk sederhanakan task.
- **Hasil:**

  | Metric | Standard | TTA |
  |--------|----------|-----|
  | mAP@50 | **0.762** | 0.743 |
  | mAP@50-95 | 0.382 | 0.374 |
  | Engaged mAP@50 | 0.853 | 0.852 |
  | NotEngaged mAP@50 | 0.670 | 0.635 |

- **Kesimpulan:** Lompatan signifikan dari plateau 54-59% ke **76.2%**. Membuktikan ambiguitas Medium adalah akar masalah. Namun dataset masih imbalanced (68:32) dan task masih detection (ada error bbox).

---

### V6 — YOLO Classify, Crop per Orang, 3-Class
- **Model:** YOLOv11s-cls
- **Resolusi:** 224px, Batch: 64, Patience: 50, label_smoothing=0.1, dropout=0.2
- **Dataset:** crops_v6 — crop per bbox dari dataset_yolo_split
  - Train: 10k/kelas (High/Low/Medium), perfectly balanced via augment offline semua kelas
  - Valid: 619–1186/kelas original | Test: 923–1705/kelas original
- **Hipotesis:** Crop per orang + balanced → atasi imbalance tanpa augment frame. Task classify lebih sederhana dari detect.
- **Hasil:**

  | Kelas | Precision | Recall | F1 |
  |-------|-----------|--------|----|
  | High | 0.768 | 0.452 | 0.569 |
  | Low | 0.594 | 0.846 | **0.698** |
  | Medium | 0.332 | 0.339 | **0.335** |
  | Overall Accuracy | — | — | **0.577** |

  Confusion matrix:
  ```
            High   Low  Medium
  High       770   522     413   ← 31% salah ke Low
  Low         30  1362     218   ← Low paling bersih
  Medium     202   408     313   ← 44% salah ke Low
  ```

- **Kesimpulan:** Medium tetap tidak bisa dipelajari (F1 0.335) bahkan dengan crop dan data balanced. Ambiguitas visual antara Medium-Low dan Medium-High terlalu besar untuk 3-class. Lanjut ke fallback 2-class.

---

### V7 — YOLO Classify, Crop per Orang, 2-Class (Pending)
- **Model:** YOLOv11s-cls
- **Resolusi:** 224px, Batch: 64, Patience: 50, dropout=0.2
- **Dataset:** crops_v7 — merge dari crops_v6
  - High + Medium original → Engaged (8353, tidak di-augment offline)
  - Low original → NotEngaged, augment offline ke 8353 (+4481)
  - Train: **8353 : 8353**, perfectly balanced
  - Valid: 1805 : 1019 | Test: 2628 : 1610 (original, tidak diaugment)
- **Hipotesis:** Kombinasi kekuatan V5 (2-class, Medium di-merge) + kekuatan V6 (crop per orang, tidak ada error bbox). Augment offline hanya Low karena Low paling konsisten secara visual (terbukti di V6 F1 0.698).
- **Hasil:**

  | Metric | Nilai |
  |--------|-------|
  | Top-1 Accuracy | **0.768** |
  | Engaged — Precision / Recall / F1 | 0.854 / 0.755 / **0.802** |
  | NotEngaged — Precision / Recall / F1 | 0.664 / 0.789 / **0.721** |
  | Macro avg F1 | **0.761** |

  Confusion matrix:
  ```
                 Engaged  NotEngaged
  Engaged           1985         643   ← 24.5% salah ke NotEngaged
  NotEngaged         340        1270   ← 21.1% salah ke Engaged
  ```

- **Kesimpulan:** V7 konsisten dengan V5 di overall (76.8% vs 76.2%) tapi **NotEngaged naik signifikan** — F1 0.721 vs mAP 0.670 di V5. Ini hasil dari dataset balanced (8353:8353) dibanding V5 yang imbalanced (68:32). Pipeline V7 lebih bersih (tidak ada error bbox) namun lebih kompleks (2 model: detect + classify).

---

## Analisis Akar Masalah (Konsolidasi)

### Mengapa Medium Selalu Rendah di V1–V6?

| Faktor | Penjelasan |
|--------|-----------|
| Ambiguitas visual | Medium secara visual overlap dengan High (siswa lihat papan sebentar) dan Low (siswa bengong sebentar) |
| Temporal nature | Engagement adalah perilaku berbasis waktu, bukan per-frame |
| Annotation inconsistency | Kasus borderline High/Medium atau Medium/Low sulit dianotasi konsisten |
| Augment frame (V4) | Satu frame berisi semua kelas — augment frame = semua kelas naik bersama |
| Augment crop (V6) | Bahkan setelah dibalance 10k/kelas via crop, Medium F1 hanya 0.335 |

### Kronologi Pendekatan dan Temuan

```
V1-V3: Model/resolusi bukan bottleneck
   ↓
V4: Augment frame-level tidak efektif (semua kelas naik bersama)
   ↓
V5: Merge Medium → breakthrough (3-class → 2-class, mAP 57%→76%)
   ↓
V6: Crop per orang tidak selamatkan Medium (F1 0.335, ambiguitas intrinsik)
   ↓
V7: Crop per orang + 2-class (gabungkan kekuatan V5 dan V6)
```

---

## Improvement yang Sudah Dicoba

| Improvement | Dicoba di | Hasil |
|-------------|-----------|-------|
| Naikkan resolusi 640→960px | V2 | +0.7% mAP, tidak signifikan |
| Naikkan resolusi 960→1280px | V3 | 0% improvement, lebih lambat |
| Upgrade model S→M | V2 | +0.7% mAP, tidak signifikan |
| SAHI untuk deteksi kecil | V2 inference | Memperburuk tracking (ID 58→131), dibuang |
| Custom BotSORT config | V2 inference | Tracking membaik (ID 58→22) ✅ |
| Balanced dataset via augment frame offline | V4 | Turun ke 54.8%, tidak efektif |
| Merge 3-class → 2-class (detection) | V5 | **+18.4% mAP → 76.2%** ✅ |
| Crop per orang + classify + balanced 10k/kelas | V6 | Top-1 Acc 57.7%, Medium tetap buruk |
| Crop per orang + 2-class + augment Low saja | V7 | Top-1 Acc 76.8%, setara V5 ✅ |
| Dataset baru (Low lebih banyak, aug ringan ~37%) | V8 | Top-1 Acc 73.0%, turun dari V7 ❌ |
| Balanced val 50:50 + AdamW + imgsz=320 | V9 | Top-1 Acc 70.6%, turun dari V7 dan V8 ❌ |

---

## Penjelasan: Perbedaan Detection (V5) vs Classification (V7)

### V5 — YOLO Detection (1 model)

```
Frame kelas (2560x1440)
         ↓
   YOLO Detect
         ↓
Output: [bbox x,y,w,h] + [kelas: Engaged/NotEngaged]
```

Model melakukan **dua hal sekaligus**: mencari posisi setiap siswa di frame (bounding box) DAN memutuskan kelasnya dalam satu forward pass.

**Konsekuensi:** Loss function gabungan — box loss + classification loss. Model harus belajar "di mana orangnya" dan "apa kelasnya" secara bersamaan. Kalau box tidak akurat, classification ikut terdampak. Dataset juga masih imbalanced (68% Engaged : 32% NotEngaged) karena augmentasi di level frame menaikkan semua kelas bersama.

---

### V7 — YOLO Detection + YOLO Classify (2 model)

```
Frame kelas (2560x1440)
         ↓
  [Model 1] YOLO Detect  ← model deteksi orang (existing)
         ↓
  Bounding box tiap siswa
         ↓
  Crop bbox dari frame
         ↓
  [Model 2] YOLO Classify ← model baru V7
         ↓
  Output: [kelas: Engaged/NotEngaged]
```

Dua tugas dipisah:
- Model 1 hanya fokus **menemukan** posisi siswa
- Model 2 hanya fokus **mengklasifikasikan** satu crop orang

**Konsekuensi:** Model classify hanya punya classification loss — tidak perlu memikirkan posisi box sama sekali. Input sudah bersih (satu orang per gambar). Dataset bisa dibalance secara tepat per kelas karena unit data adalah crop per orang, bukan frame.

---

### Perbandingan Langsung V5 vs V7

| Aspek | V5 (Detect) | V7 (Detect + Classify) |
|-------|-------------|------------------------|
| Jumlah model | 1 | 2 |
| Input classifier | Full frame | Crop per orang |
| Loss function | Box loss + Class loss | Class loss saja |
| Error sumber | Box error + class error | Class error saja |
| Dataset balance (train) | 68% : 32% | **50% : 50%** |
| Latency inferensi | Lebih cepat | +waktu crop + forward pass ke-2 |
| Kompleksitas pipeline | Sederhana | Lebih kompleks |
| Engaged | mAP 85.3% | F1 80.2% |
| NotEngaged | mAP 67.0% | F1 **72.1%** ↑ |
| Overall | mAP 76.2% | Acc **76.8%** |

> Catatan: mAP (V5) dan F1/Accuracy (V7) tidak apple-to-apple — task berbeda (detect vs classify). Tapi keduanya menunjukkan performa overall yang setara, dengan V7 unggul di NotEngaged berkat dataset balanced.

**Analogi:** V5 seperti satu dokter yang sekaligus mencari pasien dan mendiagnosa. V7 seperti dua dokter — satu khusus menemukan pasiennya, satu lagi khusus mendiagnosa.

---

---

### V8 — YOLO Classify, Crop per Orang, 2-Class, Dataset Baru
- **Model:** YOLOv11s-cls
- **Resolusi:** 224px, Batch: 64, Patience: 50, dropout=0.2
- **Dataset:** crops_v8 — crop langsung dari dataset_smp (Roboflow v13)
  - Train: **8398 : 8398**, perfectly balanced (Low 6130 original → augment offline +2268, ~37% synthetic)
  - Valid: 1805 : 1019 | Test: 2628 : 1610
- **Hipotesis:** Dataset Low lebih banyak (6130 vs 3872) dengan augmentasi lebih ringan (~37% vs ~115%) akan meningkatkan generalisasi NotEngaged.
- **Hasil:**

  | Metric | Nilai |
  |--------|-------|
  | Top-1 Accuracy (Val best) | 0.814 (epoch 6) |
  | Top-1 Accuracy (Test) | **0.730** |
  | Engaged — Precision / Recall / F1 | 0.872 / 0.661 / **0.752** |
  | NotEngaged — Precision / Recall / F1 | 0.603 / 0.841 / **0.703** |
  | Macro avg F1 | **0.727** |
  | Balanced Accuracy | 0.751 |
  | Cohen Kappa | 0.467 |
  | MCC | 0.489 |
  | ROC-AUC | **0.876** |

  Confusion matrix:
  ```
                   Engaged  NotEngaged
  Engaged             1738         890   ← 33.9% salah ke NotEngaged
  NotEngaged           256        1354   ← 15.9% salah ke Engaged
  ```

  Early stopping: best epoch **6**, training berhenti di epoch 56 (50 epoch tanpa improvement).

- **Analisis — Mengapa V8 Lebih Buruk dari V7?**

  | Aspek | V7 | V8 |
  |-------|----|----|
  | Test Accuracy | **0.768** | 0.730 |
  | Engaged Recall | **0.755** | 0.661 ↓ |
  | NotEngaged Recall | 0.789 | **0.841** ↑ |
  | Best epoch | ~pertengahan | 6 (sangat awal) |
  | Val–Test gap | 3.9% | **8.4%** ← besar |

  Tiga temuan kunci:
  1. **Model konvergen terlalu awal (epoch 6)** — optimizer MuSGD dengan lr=0.01 terlalu agresif, model overshoot dan tidak bisa recover. Val accuracy mentok di ~0.79–0.81 setelah itu.
  2. **Bias ke NotEngaged** — lebih banyak Low original di train menyebabkan model lebih "percaya diri" memprediksi NotEngaged. Recall NotEngaged naik (0.789→0.841) tapi Engaged recall turun drastis (0.755→0.661).
  3. **Val–Test gap 8.4%** — lebih besar dari V7 (3.9%), menandakan best.pt epoch 6 kurang generalize ke test set.

  ROC-AUC 0.876 sebenarnya baik (model punya kemampuan diskriminatif yang cukup), tapi threshold default 0.5 tidak optimal karena bias NotEngaged.

---

### V9 — YOLO Classify, Balanced Val (50:50), AdamW, imgsz=320
- **Model:** YOLOv11s-cls
- **Resolusi:** 320px, Batch: 128 (DDP T4×2), Patience: 50, dropout=0.2
- **Dataset:** crops_v9 — crop dari dataset_smp_balanced
  - Kelas9_2mar_1012 dipindah train→valid sehingga val seimbang 50:50
  - Train: **8398 : 8398**, perfectly balanced (Low 5336 original + 3062 augmented, ~57% synthetic)
  - Valid: 1805 : 1813 (50:50) | Test: 2628 : 1610 (62:38, tidak berubah)
- **Hipotesis:** Balanced val + AdamW lr=0.0005 + imgsz=320 akan memperbaiki bias val, konvergensi stabil, dan detail fitur lebih baik.
- **Hasil:**

  | Metric | Nilai |
  |--------|-------|
  | Top-1 Accuracy (Val best) | 0.836 (epoch 5) |
  | Top-1 Accuracy (Test) | **0.706** |
  | Engaged — Precision / Recall / F1 | 0.870 / 0.618 / **0.723** |
  | NotEngaged — Precision / Recall / F1 | 0.577 / 0.849 / **0.687** |
  | Macro avg F1 | **0.705** |
  | Balanced Accuracy | 0.734 |
  | Cohen Kappa | 0.428 |
  | MCC | 0.457 |
  | ROC-AUC | 0.859 |

  Confusion matrix:
  ```
                   Engaged  NotEngaged
  Engaged             1624        1004   ← 38.2% salah ke NotEngaged
  NotEngaged           243        1367   ← 15.1% salah ke Engaged
  ```

  Early stopping: best epoch **5**, training berhenti di epoch 55. Val-test gap: **13.0%** (membesar dari 8.4% di V8).

- **Analisis — Mengapa V9 Lebih Buruk dari V7 dan V8?**

  | Aspek | V7 | V8 | V9 |
  |-------|----|----|-----|
  | Test Accuracy | **0.768** | 0.730 | 0.706 ↓ |
  | Engaged Recall | **0.755** | 0.661 | 0.618 ↓ |
  | NotEngaged Recall | 0.789 | 0.841 | **0.849** |
  | Best epoch | ~pertengahan | 6 | 5 |
  | Val–Test gap | 3.9% | 8.4% | **13.0%** ↑ |

  Tiga temuan kunci:
  1. **Balanced val = false improvement** — val acc naik 0.814→0.836, tapi test acc justru turun ke 0.706. Val 50:50 menyembunyikan NotEngaged bias; test 62:38 (Engaged-heavy) mengeksposnya.
  2. **NotEngaged bias semakin parah** — Engaged recall turun 0.661 (V8) → 0.618 (V9). Kombinasi 57% synthetic NotEngaged di train + val session Kelas9_2mar_1012 (sesi Low murni) membuat model terlalu agresif memprediksi NotEngaged.
  3. **Val-test gap terus membesar** — 3.9% (V7) → 8.4% (V8) → 13.0% (V9). Val set semakin tidak merepresentasikan distribusi test dunia nyata (lebih banyak Engaged).

---

## Kesimpulan

1. **Ceiling ~57% untuk 3-class** berlaku di semua pendekatan — keterbatasan inheren ambiguitas visual kelas Medium.
2. **Merge ke 2-class terbukti satu-satunya breakthrough** — konsisten di V5 (mAP 76.2%) dan V7 (Acc 76.8%).
3. **V7 masih terbaik di semua metrik test** — V8 dan V9 yang mencoba berbagai perbaikan justru turun berturut-turut (76.8% → 73.0% → 70.6%).
4. **Pola kegagalan V8–V9:** NotEngaged recall naik (0.789→0.841→0.849) tapi Engaged recall turun terus (0.755→0.661→0.618). Val-test gap membesar setiap versi.
5. **Root cause utama:** Augmentasi offline agresif pada kelas NotEngaged menciptakan distribusi artificial yang tidak merepresentasikan data test. Semakin banyak synthetic NotEngaged, semakin bias model.
6. **Keputusan final model:**
   - **V7 (crop + classify)** — performa terbaik (76.8%), dataset balanced tanpa over-augmentasi, Engaged F1 terbaik (0.802)
   - **V5 (detect langsung)** — pipeline sederhana (1 model), performa overall setara V7, layak sebagai alternatif
