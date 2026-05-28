"""RetinaFace face detector via ONNX Runtime.

Uses det_10g.onnx from the InsightFace buffalo_l pack (~17 MB).
Provides the same output interface as YuNet — [x, y, w, h, confidence, 5×landmarks].

Adapted from InsightFace (MIT License).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RetinaFaceDetector:
    """RetinaFace face detector running det_10g.onnx via ONNX Runtime.

    Outputs 5-point facial landmarks compatible with face_align.norm_crop().
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        self._input_size = input_size
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._center_cache: dict = {}

        import onnxruntime as ort

        # Prefer CUDA when available (requires onnxruntime-gpu), fall back to CPU.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._sess = ort.InferenceSession(
            str(model_path),
            providers=providers,
        )

        # det_10g.onnx has 9 outputs: 3 FPN levels × (score, bbox, landmark)
        self._fmc = 3
        self._feat_strides = [8, 16, 32]
        self._num_anchors = 2

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Detect faces in an RGB image.

        Args:
            rgb: (H, W, 3) uint8 RGB image.

        Returns:
            (N, 15) array in the same format as YuNet:
                [x, y, w, h, confidence, l0x, l0y, ..., l4x, l4y]
            or None if no face is found.
        """
        img_h, img_w = rgb.shape[:2]
        in_w, in_h = self._input_size

        # --- Aspect-ratio preserving resize + pad to square ---
        im_ratio = img_h / img_w
        model_ratio = in_h / in_w
        if im_ratio > model_ratio:
            new_h = in_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = in_w
            new_h = int(new_w * im_ratio)
        det_scale = new_h / img_h

        resized = cv2.resize(rgb, (new_w, new_h))
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # --- Forward ---
        scores_list, bboxes_list, kpss_list = self._forward(canvas)

        if not scores_list:
            return None

        scores = np.vstack(scores_list)  # (total_pos, 1)
        bboxes = np.vstack(bboxes_list) / det_scale  # (total_pos, 4)  [x1,y1,x2,y2]
        kpss = np.vstack(kpss_list) / det_scale  # (total_pos, 5, 2)

        # Sort by confidence descending
        order = scores.ravel().argsort()[::-1]
        bboxes = bboxes[order]
        scores = scores[order]
        kpss = kpss[order]

        # NMS
        keep = self._nms(np.concatenate([bboxes, scores], axis=1))
        bboxes = bboxes[keep]
        scores = scores[keep]
        kpss = kpss[keep]

        if len(bboxes) == 0:
            return None

        # Convert to YuNet-compatible output:  [x, y, w, h, conf, l0x, l0y, ...]
        out = np.zeros((len(bboxes), 15), dtype=np.float32)
        out[:, 0] = bboxes[:, 0]  # x
        out[:, 1] = bboxes[:, 1]  # y
        out[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # w
        out[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # h
        out[:, 4] = scores.ravel()
        out[:, 5:15] = kpss.reshape(-1, 10)
        return out

    # ------------------------------------------------------------------ #
    # Internal: forward + decode
    # ------------------------------------------------------------------ #

    def _forward(self, img: np.ndarray) -> tuple:
        """Run the ONNX model and decode predictions at each FPN level.

        Returns:
            (scores_list, bboxes_list, kpss_list) — one entry per level
            where each entry has only the detections above threshold.
        """
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 128.0,
            self._input_size,
            (127.5, 127.5, 127.5),
            swapRB=False,  # input is already RGB
        )
        net_outs = self._sess.run(
            None, {self._sess.get_inputs()[0].name: blob}
        )

        scores_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kpss_list: List[np.ndarray] = []

        in_h = blob.shape[2]
        in_w = blob.shape[3]

        for idx, stride in enumerate(self._feat_strides):
            raw_scores = net_outs[idx]  # (num_cells×num_anchors, 1)
            bbox_preds = net_outs[idx + self._fmc] * stride  # (N, 4)
            kps_preds = net_outs[idx + self._fmc * 2] * stride  # (N, 10)

            h = in_h // stride
            w = in_w // stride
            anchors = self._get_anchor_centers(h, w, stride)

            pos = np.where(raw_scores >= self._conf_threshold)[0]
            if len(pos) == 0:
                continue

            bboxes = self._distance2bbox(anchors, bbox_preds)
            kpss = (
                self._distance2kps(anchors, kps_preds)
                .reshape(-1, 5, 2)
            )

            scores_list.append(raw_scores[pos])
            bboxes_list.append(bboxes[pos])
            kpss_list.append(kpss[pos])

        return scores_list, bboxes_list, kpss_list

    # ------------------------------------------------------------------ #
    # Anchor helpers
    # ------------------------------------------------------------------ #

    def _get_anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        """Cached anchor-centre grid for a given (height, width, stride)."""
        key = (h, w, stride)
        if key in self._center_cache:
            return self._center_cache[key]

        # mgrid[::-1] gives (x, y) ordering
        centers = np.stack(
            np.mgrid[:h, :w][::-1], axis=-1
        ).astype(np.float32)
        centers = (centers * stride).reshape(-1, 2)

        if self._num_anchors > 1:
            centers = np.stack(
                [centers] * self._num_anchors, axis=1
            ).reshape(-1, 2)

        if len(self._center_cache) < 100:
            self._center_cache[key] = centers
        return centers

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #

    @staticmethod
    def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """Decode distance prediction to [x1, y1, x2, y2] bbox."""
        return np.stack(
            [
                points[:, 0] - distance[:, 0],
                points[:, 1] - distance[:, 1],
                points[:, 0] + distance[:, 2],
                points[:, 1] + distance[:, 3],
            ],
            axis=-1,
        )

    @staticmethod
    def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """Decode distance prediction to landmark coordinates.

        All 5 landmarks are relative to the anchor centre (points[:, 0:2]).
        """
        parts = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            parts.append(px[:, None])
            parts.append(py[:, None])
        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------ #
    # NMS
    # ------------------------------------------------------------------ #

    def _nms(self, dets: np.ndarray) -> np.ndarray:
        """Standard NMS — keeps indices of remaining detections."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[np.where(ovr <= self._nms_threshold)[0] + 1]

        return np.array(keep)
