# Added by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add utils for evaluation


from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class SequenceMetric:
    sequence_name: str
    precision_score: np.ndarray
    success_score: np.ndarray
    normalized_precision_score: Optional[np.ndarray] = None
    frame_num: Optional[int] = None


class DatasetMetrics:
    def __init__(self, compute_npr=False, norm='sequence'):
        self._compute_npr = compute_npr
        self._norm_method = norm
        self._sequence_metrics: List[SequenceMetric] = []

    def add_metric(self, metric: SequenceMetric) -> None:
        self._sequence_metrics.append(metric)

    def _check_empty(self) -> None:
        if not self._sequence_metrics:
            raise ValueError("No sequence metrics have been added yet")

    def compute(self):
        self._check_empty()

        npr_score = 0.
        pr_score = 0.
        sr_score = 0.
        total_frame_num = 0

        for metric in self._sequence_metrics:
            pr_score += metric.precision_score
            sr_score += metric.success_score

            if self._compute_npr:
                npr_score += metric.normalized_precision_score

            if metric.frame_num is not None:
                total_frame_num += metric.frame_num

        if self._norm_method == 'sequence':
            total_count = len(self._sequence_metrics)
            pr_score /= total_count
            sr_score /= total_count

            if self._compute_npr:
                npr_score /= total_count

        elif self._norm_method == 'frame_num':
            pr_score = pr_score / total_frame_num
            sr_score = sr_score / total_frame_num

        else:
            raise NotImplementedError

        res = {
            "success_score": sr_score,
            "precision_score": pr_score,
        }

        if self._compute_npr:
            res["norm_precision"] = npr_score

        return res


def cle(rect1, rect2, strict=True):
    if strict:
        cp1 = [(rect1[2] - 1) / 2. + rect1[0], (rect1[3] - 1) / 2. + rect1[1]]
        cp2 = [(rect2[2] - 1) / 2. + rect2[0], (rect2[3] - 1) / 2. + rect2[1]]
    else:
        cp1 = [rect1[2] / 2. + rect1[0], rect1[3] / 2. + rect1[1]]
        cp2 = [rect2[2] / 2. + rect2[0], rect2[3] / 2. + rect2[1]]
    d = ((cp1[0] - cp2[0]) ** 2 + (cp1[1] - cp2[1]) ** 2) ** 0.5
    return d


def normalize(cx, cy, gt_w, gt_h, eps=1e-8):
    return cx / (gt_w + eps), cy / (gt_h + eps)


def normalize_cle(rect1, rect2, strict=True):
    if strict:
        cp1 = [(rect1[2] - 1) / 2. + rect1[0], (rect1[3] - 1) / 2. + rect1[1]]
        cp2 = [(rect2[2] - 1) / 2. + rect2[0], (rect2[3] - 1) / 2. + rect2[1]]
    else:
        cp1 = [rect1[2] / 2. + rect1[0], rect1[3] / 2. + rect1[1]]
        cp2 = [rect2[2] / 2. + rect2[0], rect2[3] / 2. + rect2[1]]
    cp1 = normalize(cp1[0], cp1[1], rect2[2], rect2[3])
    cp2 = normalize(cp2[0], cp2[1], rect2[2], rect2[3])
    d = ((cp1[0] - cp2[0]) ** 2 + (cp1[1] - cp2[1]) ** 2) ** 0.5
    return d


def iou(rect1, rect2):
    x1, y1, x2, y2 = ltwh_2_ltrb(rect1)
    tx1, ty1, tx2, ty2 = ltwh_2_ltrb(rect2)

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1 + 1)
    hh = np.maximum(0, yy2 - yy1 + 1)

    area = rect1[-1] * rect1[-2]
    target_a = rect2[-1] * rect2[-2]
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou


def ltwh_2_ltrb(rect):
    return [rect[0], rect[1], rect[2] + rect[0] - 1, rect[3] + rect[1] - 1]


def serial_process(fun, *serial):
    return list(map(fun, *serial))
