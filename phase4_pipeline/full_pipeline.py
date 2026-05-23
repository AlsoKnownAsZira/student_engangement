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
import time as _time
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
        min_box_area_frac: float = 0.001,
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
        self.metrics = EngagementMetrics()
        self.tracking_data: list[dict] = []
        self._timing: dict[str, list[float]] = {'detect': [], 'classify': []}

        self.logger.info(
            f"Pipeline ready (V10 2-stage) — stride={self.frame_stride}, "
            f"thr={self.classify_threshold}"
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
        """
        Process a video in two passes:
          Pass 1 — inference: detect + classify every sampled frame, fill tracking_data.
          Pass 2 — render: re-read video and draw annotations with post-merged track IDs.
        Two-pass ensures the video displays the same IDs shown in the analysis results.
        """
        video_path = str(video_path)
        self.logger.info(f"Processing: {video_path}")

        self.metrics.reset()
        self.tracking_data = []
        self._timing = {'detect': [], 'classify': []}

        # ── Pass 1: Inference ──────────────────────────────────────────────
        src_fps, width, height, n_sampled = self._inference_pass(
            video_path, show_preview, limit_frames
        )
        self.logger.info(f"Pass 1 done — sampled {n_sampled} frames")

        # ── Post-processing ────────────────────────────────────────────────
        df = pd.DataFrame(self.tracking_data)
        if len(df) > 0:
            df = self._merge_tracks(df)
            self.tracking_data = df.to_dict('records')

        if save_csv and output_path and len(df) > 0:
            csv_path = Path(output_path).with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved CSV: {csv_path}")

        # ── Pass 2: Render with merged IDs ─────────────────────────────────
        if output_path and len(df) > 0:
            self._render_pass(video_path, str(output_path), df, src_fps, width, height, n_sampled)
            self.logger.info(f"Pass 2 done — video written to {output_path}")

        return df

    def _inference_pass(
        self,
        video_path: str,
        show_preview: bool,
        limit_frames: int | None,
    ) -> tuple[float, int, int, int]:
        """
        Pass 1: Run detection + classification on sampled frames.
        Fills self.tracking_data. Does NOT write video.
        Returns (src_fps, width, height, n_sampled).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 15
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(
            f"Source: {width}x{height} @ {src_fps:.1f}fps, {total} frames | "
            f"Effective: {max(1.0, src_fps / self.frame_stride):.1f}fps (stride={self.frame_stride})"
        )

        src_idx     = 0
        sampled_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if src_idx % self.frame_stride != 0:
                    src_idx += 1
                    continue

                if limit_frames and sampled_idx >= limit_frames:
                    self.logger.info(f"Reached frame limit: {limit_frames}")
                    break

                if sampled_idx % 30 == 0:
                    self.logger.info(f"Frame {src_idx}/{total} (sampled #{sampled_idx})...")

                # Stage 1: detect + track
                _t_det = _time.perf_counter()
                det_results = self.detector.track(
                    source=frame,
                    tracker=self.tracker_config,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    persist=True,
                    verbose=False,
                )
                self._timing['detect'].append((_time.perf_counter() - _t_det) * 1000)
                det_result = det_results[0] if det_results else None

                frame_scores: dict[int, tuple[float, str]] = {}
                untracked_counter = 99000
                frame_area = frame.shape[0] * frame.shape[1]

                valid: list[dict] = []
                if det_result is not None and det_result.boxes is not None:
                    for box in det_result.boxes:
                        tid = int(box.id[0]) if box.id is not None else (untracked_counter := untracked_counter + 1)
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        det_conf = float(box.conf[0])

                        if (x2 - x1) * (y2 - y1) / frame_area < self.min_box_area_frac:
                            continue
                        cx1 = max(0, x1); cy1 = max(0, y1)
                        cx2 = min(width, x2); cy2 = min(height, y2)
                        if cx2 <= cx1 or cy2 <= cy1:
                            continue

                        valid.append({
                            'track_id': tid,
                            'bbox': (x1, y1, x2, y2),
                            'det_conf': det_conf,
                            'crop': frame[cy1:cy2, cx1:cx2],
                        })

                # Stage 2: classify
                if valid:
                    _t_cls = _time.perf_counter()
                    probs_engaged = self._classify_crops([v['crop'] for v in valid])
                    self._timing['classify'].append((_time.perf_counter() - _t_cls) * 1000)
                else:
                    self._timing['classify'].append(0.0)
                    probs_engaged = []

                for v, p_eng in zip(valid, probs_engaged):
                    tid   = v['track_id']
                    label = self._label_from_prob(p_eng)
                    conf  = p_eng if label == self.LEVEL_ENGAGED else (1.0 - p_eng)
                    frame_scores[tid] = (conf, label)
                    self.tracking_data.append({
                        'frame': sampled_idx,
                        'source_frame': src_idx,
                        'track_id': tid,
                        'x1': v['bbox'][0], 'y1': v['bbox'][1],
                        'x2': v['bbox'][2], 'y2': v['bbox'][3],
                        'detection_conf': v['det_conf'],
                        'prob_engaged': round(p_eng, 4),
                        'engagement_level': label,
                        'engagement_score': round(conf, 4),
                    })

                self.metrics.add_frame(frame_scores)

                if show_preview:
                    preview = frame.copy()
                    for v, p_eng in zip(valid, probs_engaged):
                        tid   = v['track_id']
                        label = self._label_from_prob(p_eng)
                        conf  = p_eng if label == self.LEVEL_ENGAGED else (1.0 - p_eng)
                        self._draw_person(preview, tid, v['bbox'], label, conf, v['det_conf'])
                    self._draw_summary(preview, frame_scores, sampled_idx)
                    cv2.imshow('Pipeline V10 — Pass 1 (inference)', preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Preview stopped by user")
                        break

                sampled_idx += 1
                src_idx += 1

        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()

        return src_fps, width, height, sampled_idx

    def _render_pass(
        self,
        video_path: str,
        output_path: str,
        df: pd.DataFrame,
        src_fps: float,
        width: int,
        height: int,
        n_sampled: int,
    ) -> None:
        """
        Pass 2: Re-read source video and render annotations using merged track IDs.
        Produces a video where IDs are consistent with the analysis results.
        """
        self.logger.info(f"Pass 2 — rendering {n_sampled} frames with merged IDs...")

        # Build per-frame lookup from merged tracking data.
        # If the same merged ID appears twice in a frame (edge case from temporal tolerance),
        # keep only the detection with the highest engagement_score.
        frame_lookup: dict[int, dict[int, dict]] = {}
        for row in df.itertuples(index=False):
            fid = int(row.frame)
            tid = int(row.track_id)
            det = {
                'track_id': tid,
                'bbox':     (int(row.x1), int(row.y1), int(row.x2), int(row.y2)),
                'label':    row.engagement_level,
                'score':    float(row.engagement_score),
                'det_conf': float(row.detection_conf),
            }
            if fid not in frame_lookup or tid not in frame_lookup[fid]:
                frame_lookup.setdefault(fid, {})[tid] = det
            elif det['score'] > frame_lookup[fid][tid]['score']:
                frame_lookup[fid][tid] = det

        out_fps = max(1.0, src_fps / self.frame_stride)
        writer  = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            out_fps,
            (width, height),
        )

        cap = cv2.VideoCapture(video_path)
        src_idx     = 0
        sampled_idx = 0

        try:
            while sampled_idx < n_sampled:
                ret, frame = cap.read()
                if not ret:
                    break

                if src_idx % self.frame_stride != 0:
                    src_idx += 1
                    continue

                detections  = frame_lookup.get(sampled_idx, {})
                frame_scores: dict[int, tuple[float, str]] = {}

                for d in detections.values():
                    self._draw_person(frame, d['track_id'], d['bbox'], d['label'], d['score'], d['det_conf'])
                    frame_scores[d['track_id']] = (d['score'], d['label'])

                self._draw_summary(frame, frame_scores, sampled_idx)
                writer.write(frame)

                sampled_idx += 1
                src_idx += 1

        finally:
            cap.release()
            writer.release()

    # ═══════════════════════════════════════════════════════════════════════
    # Drawing
    # ═══════════════════════════════════════════════════════════════════════

    def _draw_person(self, frame, track_id, bbox, level, score, det_conf: float = 1.0):
        x1, y1, x2, y2 = map(int, bbox)
        color = self.COLORS.get(level, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"ID:{track_id} | {level.upper()}"
        label2 = f"Cls:{score:.2f}  Det:{det_conf:.2f}"

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
    # Post-processing: track merging
    # ═══════════════════════════════════════════════════════════════════════

    def _merge_tracks(self, df: pd.DataFrame, dist_thresh: float = 80.0) -> pd.DataFrame:
        """
        Merge sequential track segments that appear at the same spatial position.

        Students sit in fixed seats — if the tracker loses a student and then
        re-detects them, BotSORT assigns a new ID even though it is the same
        physical person.  This function uses the domain knowledge that
        (a) seats are fixed and (b) two tracks at the same position cannot
        overlap in time to merge those fragments into a single canonical ID.

        Only merges tracks with NO temporal overlap (±2 frame tolerance).
        Spatial threshold default 100 px works well for typical CCTV resolutions.
        """
        if df.empty or df['track_id'].nunique() <= 1:
            return df

        # Per-track: median centroid and temporal range
        stats: dict[int, dict] = {}
        for tid, g in df.groupby('track_id'):
            stats[tid] = {
                'cx': ((g['x1'] + g['x2']) / 2).median(),
                'cy': ((g['y1'] + g['y2']) / 2).median(),
                'fmin': int(g['frame'].min()),
                'fmax': int(g['frame'].max()),
            }

        track_ids = sorted(stats.keys())

        # Union-Find with path compression
        parent = {tid: tid for tid in track_ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                if rx < ry:
                    parent[ry] = rx
                else:
                    parent[rx] = ry

        TEMPORAL_TOL = 2  # frames — tolerate tiny overlaps from tracker jitter

        for i, a in enumerate(track_ids):
            for b in track_ids[i + 1:]:
                sa, sb = stats[a], stats[b]
                # Skip if they overlap in time (same person can't be in two places)
                if sa['fmax'] - TEMPORAL_TOL >= sb['fmin'] and sb['fmax'] - TEMPORAL_TOL >= sa['fmin']:
                    continue
                # Merge if centroids are within dist_thresh pixels
                dist = ((sa['cx'] - sb['cx']) ** 2 + (sa['cy'] - sb['cy']) ** 2) ** 0.5
                if dist < dist_thresh:
                    union(a, b)

        canonical_map = {tid: find(tid) for tid in track_ids}
        n_before = len(track_ids)
        n_after  = len(set(canonical_map.values()))
        self.logger.info(
            f"Track merge: {n_before} segments -> {n_after} students "
            f"(dist_thresh={dist_thresh:.0f}px)"
        )

        # Remap canonical IDs to sequential 1-based integers ordered by first appearance.
        # This ensures the video shows clean IDs (1, 2, 3 ...) instead of raw tracker IDs.
        first_frame: dict[int, int] = {}
        for tid in track_ids:
            canon = canonical_map[tid]
            f = stats[tid]['fmin']
            if canon not in first_frame or f < first_frame[canon]:
                first_frame[canon] = f

        sorted_canonicals = sorted(first_frame, key=first_frame.get)
        sequential = {c: i + 1 for i, c in enumerate(sorted_canonicals)}
        id_map = {tid: sequential[canonical_map[tid]] for tid in track_ids}

        df = df.copy()
        df['track_id'] = df['track_id'].map(id_map)
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # Stats
    # ═══════════════════════════════════════════════════════════════════════

    def get_statistics(self) -> dict | None:
        if not self.tracking_data:
            return None
        df = pd.DataFrame(self.tracking_data)
        stats: dict = {
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
        n = len(self._timing['detect'])
        if n > 0:
            avg_det = sum(self._timing['detect']) / n
            avg_cls = sum(self._timing['classify']) / n
            avg_total = avg_det + avg_cls
            stats['avg_detector_ms'] = round(avg_det, 2)
            stats['avg_classifier_ms'] = round(avg_cls, 2)
            stats['avg_pipeline_ms_per_frame'] = round(avg_total, 2)
        return stats


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
