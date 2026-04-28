"""
Phase 4: Full Pipeline — 2-MODEL ARCHITECTURE (V10)

Pipeline:
    Frame (sampled @ frame_stride)
        -> [Model 1] YOLO Detector (bbox per orang, ID via BotSORT)
        -> Crop tiap bbox (224px)
        -> [Model 2] YOLO Classifier (prob_engaged)
        -> Threshold (default 0.170, dari per-session calibration V10)
        -> EngagementSmoother (temporal voting per track)
        -> Annotated frame + CSV

Class output: 2-class only — `engaged` / `not-engaged`.

Frame sampling:
    Source video QHD @ 15 fps biasanya tidak perlu inferensi tiap frame.
    `frame_stride=5` = ambil 1 dari 5 frame -> efektif 3 fps.
    Output video di-render pada efektif fps (lebih kecil, lebih cepat di-upload).

Usage:
    pipeline = TwoStagePipeline(
        detector_model='models/best_v5.pt',
        classifier_model='models/best_v10.pt',
        classify_threshold=0.170,
        frame_stride=5,
    )
    df = pipeline.process_video(video_path, output_path='out.mp4')
"""

from __future__ import annotations

import sys
sys.path.append('..')

import cv2
import numpy as np
from pathlib import Path
import argparse
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO

import config  # noqa: F401  -- kept for backward-compat with project layout
from utils.video_utils import VideoReader, VideoWriter  # noqa: F401
from utils.metrics import EngagementMetrics
from utils.logger import setup_logger


class EngagementSmoother:
    """Confidence-weighted majority voting over a sliding window per track_id."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, track_id: int, level: str, confidence: float) -> None:
        self.history[track_id].append((level, confidence))

    def get_smoothed(self, track_id: int) -> tuple[str | None, float]:
        if not self.history.get(track_id):
            return None, 0.0
        votes: dict[str, float] = defaultdict(float)
        for lv, c in self.history[track_id]:
            votes[lv] += c
        best = max(votes, key=votes.get)
        avg_conf = votes[best] / len(self.history[track_id])
        return best, avg_conf

    def cleanup_stale(self, active_ids: set[int]) -> None:
        for tid in [t for t in self.history if t not in active_ids]:
            del self.history[tid]


class TwoStagePipeline:
    """
    Detector + Classifier pipeline (V10).

    Detection model is used purely for *bounding boxes + tracking IDs*. Its
    class output is ignored — classification comes from the second model.
    """

    # 2-class color scheme
    COLORS = {
        'engaged':     (0, 200, 0),       # Green
        'not-engaged': (0, 0, 220),       # Red
    }

    LEVEL_ENGAGED = 'engaged'
    LEVEL_NOT_ENGAGED = 'not-engaged'

    def __init__(
        self,
        detector_model: str = 'models/best_v5.pt',
        classifier_model: str = 'models/best_v10.pt',
        classify_threshold: float = 0.170,
        tracker_config: str = 'custom_botsort.yaml',
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.5,
        device: int | str = 0,
        smoothing_window: int = 10,
        frame_stride: int = 5,
        classifier_imgsz: int = 224,
        min_box_area_frac: float = 0.003,
    ):
        self.logger = setup_logger(self.__class__.__name__)

        # ── Models ────────────────────────────────────────────────────────
        self.logger.info(f"Loading DETECTOR : {detector_model}")
        self.detector = YOLO(detector_model)
        if self.detector.task != 'detect':
            self.logger.warning(
                f"Detector model task is '{self.detector.task}', expected 'detect'. "
                "Pipeline may fail."
            )

        self.logger.info(f"Loading CLASSIFIER: {classifier_model}")
        self.classifier = YOLO(classifier_model)
        if self.classifier.task != 'classify':
            self.logger.warning(
                f"Classifier model task is '{self.classifier.task}', expected 'classify'."
            )

        # Resolve "Engaged" class index in classifier (case-insensitive)
        cls_names = getattr(self.classifier, 'names', {}) or {}
        self._engaged_idx = next(
            (k for k, v in cls_names.items() if str(v).lower().startswith('engag')),
            0,
        )
        self.logger.info(
            f"Classifier classes: {cls_names} -> 'engaged' index = {self._engaged_idx}"
        )

        # ── Tracker config (resolve relative path) ────────────────────────
        tracker_path = Path(tracker_config)
        if not tracker_path.is_absolute() and not tracker_path.exists():
            local = Path(__file__).parent / tracker_config
            if local.exists():
                tracker_config = str(local)
        self.tracker_config = tracker_config

        # ── Hyperparams ───────────────────────────────────────────────────
        self.classify_threshold = classify_threshold
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.frame_stride = max(1, int(frame_stride))
        self.classifier_imgsz = classifier_imgsz
        self.min_box_area_frac = min_box_area_frac

        # ── State ─────────────────────────────────────────────────────────
        self.smoother = EngagementSmoother(window_size=smoothing_window)
        self.metrics = EngagementMetrics()
        self.tracking_data: list[dict] = []

        self.logger.info(
            f"Pipeline ready (V10 2-stage) — stride={self.frame_stride}, "
            f"thr={self.classify_threshold}, smooth={smoothing_window}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Classifier helper
    # ═══════════════════════════════════════════════════════════════════════

    def _classify_crops(self, crops: list[np.ndarray]) -> list[float]:
        """Run classifier on a batch of BGR crops. Returns list of P(engaged)."""
        if not crops:
            return []
        results = self.classifier.predict(
            crops,
            imgsz=self.classifier_imgsz,
            device=self.device,
            verbose=False,
        )
        probs: list[float] = []
        for r in results:
            data = r.probs.data.cpu().numpy()
            probs.append(float(data[self._engaged_idx]))
        return probs

    def _label_from_prob(self, p_engaged: float) -> str:
        return self.LEVEL_ENGAGED if p_engaged >= self.classify_threshold else self.LEVEL_NOT_ENGAGED

    # ═══════════════════════════════════════════════════════════════════════
    # Main entry point
    # ═══════════════════════════════════════════════════════════════════════

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        save_csv: bool = True,
        show_preview: bool = False,
        limit_frames: int | None = None,
    ) -> pd.DataFrame:
        """Process a video and return per-detection DataFrame."""
        video_path = str(video_path)
        self.logger.info(f"Processing: {video_path}")

        self.metrics.reset()
        self.tracking_data = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 15
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = max(1.0, src_fps / self.frame_stride)
        self.logger.info(
            f"Source: {width}x{height} @ {src_fps:.1f}fps, {total} frames | "
            f"Output: {out_fps:.1f}fps (stride={self.frame_stride})"
        )

        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                out_fps,
                (width, height),
            )
            self.logger.info(f"Output: {output_path}")

        src_idx = 0          # raw frame index in source video
        sampled_idx = 0      # index of frames actually inferenced

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames per stride
                if src_idx % self.frame_stride != 0:
                    src_idx += 1
                    continue

                if limit_frames and sampled_idx >= limit_frames:
                    self.logger.info(f"Reached frame limit: {limit_frames}")
                    break

                if sampled_idx % 30 == 0:
                    self.logger.info(
                        f"Frame {src_idx}/{total} (sampled #{sampled_idx})..."
                    )

                # ── Stage 1: detection + tracking ─────────────────────────
                det_results = self.detector.track(
                    source=frame,
                    tracker=self.tracker_config,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    persist=True,
                    verbose=False,
                )
                det_result = det_results[0] if det_results else None

                frame_scores: dict[int, tuple[float, str]] = {}
                untracked_counter = 99000
                frame_area = frame.shape[0] * frame.shape[1]

                # Collect valid bboxes + crops
                valid: list[dict] = []
                if det_result is not None and det_result.boxes is not None:
                    for box in det_result.boxes:
                        if box.id is not None:
                            tid = int(box.id[0])
                        else:
                            tid = untracked_counter
                            untracked_counter += 1

                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        det_conf = float(box.conf[0])

                        # Filter by min area
                        if (x2 - x1) * (y2 - y1) / frame_area < self.min_box_area_frac:
                            continue

                        # Sanity-clip crop
                        cx1 = max(0, x1); cy1 = max(0, y1)
                        cx2 = min(frame.shape[1], x2); cy2 = min(frame.shape[0], y2)
                        if cx2 <= cx1 or cy2 <= cy1:
                            continue

                        crop = frame[cy1:cy2, cx1:cx2]
                        valid.append({
                            'track_id': tid,
                            'bbox': (x1, y1, x2, y2),
                            'det_conf': det_conf,
                            'crop': crop,
                        })

                # ── Stage 2: classify all crops in one batched call ───────
                if valid:
                    probs_engaged = self._classify_crops([v['crop'] for v in valid])
                else:
                    probs_engaged = []

                for v, p_eng in zip(valid, probs_engaged):
                    tid = v['track_id']
                    raw_label = self._label_from_prob(p_eng)
                    raw_conf = p_eng if raw_label == self.LEVEL_ENGAGED else (1.0 - p_eng)

                    self.smoother.update(tid, raw_label, raw_conf)
                    smoothed_label, smoothed_conf = self.smoother.get_smoothed(tid)
                    if smoothed_label is None:
                        smoothed_label, smoothed_conf = raw_label, raw_conf

                    frame_scores[tid] = (smoothed_conf, smoothed_label)

                    self.tracking_data.append({
                        'frame': sampled_idx,
                        'source_frame': src_idx,
                        'track_id': tid,
                        'x1': v['bbox'][0], 'y1': v['bbox'][1],
                        'x2': v['bbox'][2], 'y2': v['bbox'][3],
                        'detection_conf': v['det_conf'],
                        'prob_engaged': round(p_eng, 4),
                        'raw_engagement': raw_label,
                        'engagement_level': smoothed_label,
                        'engagement_score': round(smoothed_conf, 4),
                    })

                    self._draw_person(frame, tid, v['bbox'], smoothed_label, smoothed_conf)

                # Cleanup smoother for IDs gone for too long
                self.smoother.cleanup_stale(set(frame_scores.keys()))
                self.metrics.add_frame(frame_scores)
                self._draw_summary(frame, frame_scores, sampled_idx)

                if show_preview:
                    cv2.imshow('Pipeline V10 Preview', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Preview stopped by user")
                        break

                if writer:
                    writer.write(frame)

                sampled_idx += 1
                src_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        self.logger.info(
            f"Done. Source frames read: {src_idx}, inferenced: {sampled_idx}"
        )

        df = pd.DataFrame(self.tracking_data)
        if save_csv and output_path and len(df) > 0:
            csv_path = Path(output_path).with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved CSV: {csv_path}")

        return df

    # ═══════════════════════════════════════════════════════════════════════
    # Drawing
    # ═══════════════════════════════════════════════════════════════════════

    def _draw_person(self, frame, track_id, bbox, level, score):
        x1, y1, x2, y2 = map(int, bbox)
        color = self.COLORS.get(level, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"ID:{track_id} | {level.upper()}"
        label2 = f"Conf: {score:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (l2w, l2h), _ = cv2.getTextSize(label2, font, font_scale, thickness)

        cv2.rectangle(
            frame,
            (x1, y1 - lh - l2h - 15),
            (x1 + max(lw, l2w) + 10, y1),
            color, -1,
        )
        cv2.putText(frame, label, (x1 + 5, y1 - l2h - 10),
                    font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, label2, (x1 + 5, y1 - 5),
                    font, font_scale, (255, 255, 255), thickness)

    def _draw_summary(self, frame, frame_scores, frame_idx):
        if not frame_scores:
            return

        counts = {self.LEVEL_ENGAGED: 0, self.LEVEL_NOT_ENGAGED: 0}
        for _, lv in frame_scores.values():
            if lv in counts:
                counts[lv] += 1

        total = len(frame_scores)
        eng = counts[self.LEVEL_ENGAGED]
        ne = counts[self.LEVEL_NOT_ENGAGED]
        lines = [
            f"Frame: {frame_idx}",
            f"Students: {total}",
            f"Engaged: {eng} ({eng/total*100:.0f}%)" if total else "Engaged: 0",
            f"Not Engaged: {ne} ({ne/total*100:.0f}%)" if total else "Not Engaged: 0",
        ]

        panel_h = len(lines) * 35 + 20
        panel_w = 320
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y = 40
        for i, line in enumerate(lines):
            if i >= 2:
                key = self.LEVEL_ENGAGED if i == 2 else self.LEVEL_NOT_ENGAGED
                cv2.circle(frame, (25, y - 8), 8, self.COLORS[key], -1)
                cv2.putText(frame, line, (45, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, line, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 35

    # ═══════════════════════════════════════════════════════════════════════
    # Stats
    # ═══════════════════════════════════════════════════════════════════════

    def get_statistics(self) -> dict | None:
        if not self.tracking_data:
            return None
        df = pd.DataFrame(self.tracking_data)
        return {
            'total_frames': df['frame'].nunique(),
            'total_detections': len(df),
            'unique_students': df['track_id'].nunique(),
            'engagement_distribution': df['engagement_level'].value_counts().to_dict(),
            'avg_confidence': float(df['engagement_score'].mean()),
            'avg_students_per_frame': len(df) / max(1, df['frame'].nunique()),
            'pipeline': '2-stage (detect + classify V10)',
            'classify_threshold': self.classify_threshold,
            'frame_stride': self.frame_stride,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Backward-compat alias
# Existing code imports `FullPipeline` — keep that name working.
# ═══════════════════════════════════════════════════════════════════════════════

class FullPipeline(TwoStagePipeline):
    """Backward-compat shim: maps old kwargs to TwoStagePipeline.

    Old call:
        FullPipeline(main_model=..., tracker_config=..., conf_threshold=...,
                     use_sahi=..., sahi_slice_size=..., sahi_overlap=...)

    Now interpreted as:
        TwoStagePipeline(detector_model=main_model, ...)

    SAHI args are accepted but ignored (V10 pipeline doesn't use SAHI).
    """

    def __init__(
        self,
        main_model: str | None = None,
        detector_model: str | None = None,
        classifier_model: str = 'models/best_v10.pt',
        classify_threshold: float = 0.170,
        tracker_config: str = 'custom_botsort.yaml',
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.5,
        device: int | str = 0,
        smoothing_window: int = 10,
        frame_stride: int = 5,
        # SAHI kwargs accepted for backward-compat but ignored
        use_sahi: bool = False,
        sahi_slice_size: int = 640,
        sahi_overlap: float = 0.2,
        **_ignored,
    ):
        if use_sahi:
            # Pipeline V10 doesn't ship a SAHI path; warn and continue.
            setup_logger(self.__class__.__name__).warning(
                "use_sahi=True ignored — V10 pipeline runs detect+classify without SAHI."
            )
        det = detector_model or main_model or 'models/best_v5.pt'
        super().__init__(
            detector_model=det,
            classifier_model=classifier_model,
            classify_threshold=classify_threshold,
            tracker_config=tracker_config,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            smoothing_window=smoothing_window,
            frame_stride=frame_stride,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Phase 4 V10: 2-stage detect + classify pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--detector', type=str, default='models/best_v5.pt')
    parser.add_argument('--classifier', type=str, default='models/best_v10.pt')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', default=0)
    parser.add_argument('--conf', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0.170,
                        help='P(engaged) threshold (default: 0.170 from V10 calibration)')
    parser.add_argument('--stride', type=int, default=5,
                        help='Process every Nth frame (default 5 = 3fps from 15fps source)')
    parser.add_argument('--smoothing', type=int, default=10)
    parser.add_argument('--tracker', type=str, default='custom_botsort.yaml')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    pipeline = TwoStagePipeline(
        detector_model=args.detector,
        classifier_model=args.classifier,
        classify_threshold=args.threshold,
        tracker_config=args.tracker,
        conf_threshold=args.conf,
        device=args.device,
        smoothing_window=args.smoothing,
        frame_stride=args.stride,
    )

    df = pipeline.process_video(
        args.video,
        output_path=args.output,
        save_csv=True,
        show_preview=args.preview,
        limit_frames=args.limit,
    )

    logger = setup_logger("main")
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)

    stats = pipeline.get_statistics()
    if stats:
        for k, v in stats.items():
            logger.info(f"{k}: {v}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
