from itertools import groupby
from operator import itemgetter
from typing import Tuple

import numpy as np


def lr2cw(bbox_lr: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from left-right (LR) to center-width (CW) format.

    :param bbox_lr: LR bounding boxes. Sized [N, 2].
    :return: CW bounding boxes. Sized [N, 2].
    """
    bbox_lr = np.asarray(bbox_lr, dtype=np.float32).reshape((-1, 2))
    center = (bbox_lr[:, 0] + bbox_lr[:, 1]) / 2
    width = bbox_lr[:, 1] - bbox_lr[:, 0]
    bbox_cw = np.vstack((center, width)).T
    return bbox_cw


def cw2lr(bbox_cw: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from center-width (CW) to left-right (LR) format.

    :param bbox_cw: CW bounding boxes. Sized [N, 2].
    :return: LR bounding boxes. Sized [N, 2].
    """
    bbox_cw = np.asarray(bbox_cw, dtype=np.float32).reshape((-1, 2))
    left = bbox_cw[:, 0] - bbox_cw[:, 1] / 2
    right = bbox_cw[:, 0] + bbox_cw[:, 1] / 2
    bbox_lr = np.vstack((left, right)).T
    return bbox_lr


def seq2bbox(sequence: np.ndarray) -> np.ndarray:
    """Generate CW bbox from binary sequence mask"""
    sequence = np.asarray(sequence, dtype=bool)
    selected_indices, = np.where(sequence == 1)

    bboxes_lr = []
    for k, g in groupby(enumerate(selected_indices), lambda x: x[0] - x[1]):
        segment = list(map(itemgetter(1), g))
        start_frame, end_frame = segment[0], segment[-1] + 1
        bboxes_lr.append([start_frame, end_frame])

    bboxes_lr = np.asarray(bboxes_lr, dtype=np.int32)
    return bboxes_lr


def iou_lr(anchor_bbox: np.ndarray, target_bbox: np.ndarray) -> np.ndarray:
    """Compute iou between multiple LR bbox pairs.

    :param anchor_bbox: LR anchor bboxes. Sized [N, 2].
    :param target_bbox: LR target bboxes. Sized [N, 2].
    :return: IoU between each bbox pair. Sized [N].
    """
    anchor_left, anchor_right = anchor_bbox[:, 0], anchor_bbox[:, 1]
    target_left, target_right = target_bbox[:, 0], target_bbox[:, 1]

    inter_left = np.maximum(anchor_left, target_left)
    inter_right = np.minimum(anchor_right, target_right)
    union_left = np.minimum(anchor_left, target_left)
    union_right = np.maximum(anchor_right, target_right)

    intersect = inter_right - inter_left
    intersect[intersect < 0] = 0
    union = union_right - union_left
    union[union <= 0] = 1e-6

    iou = intersect / union
    return iou


def iou_cw(anchor_bbox: np.ndarray, target_bbox: np.ndarray) -> np.ndarray:
    """Compute iou between multiple CW bbox pairs. See ``iou_lr``"""
    anchor_bbox_lr = cw2lr(anchor_bbox)
    target_bbox_lr = cw2lr(target_bbox)
    return iou_lr(anchor_bbox_lr, target_bbox_lr)


def nms(scores: np.ndarray,
        bboxes: np.ndarray,
        thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """Non-Maximum Suppression (NMS) algorithm on 1D bbox.

    :param scores: List of confidence scores for bboxes. Sized [N].
    :param bboxes: List of LR bboxes. Sized [N, 2].
    :param thresh: IoU threshold. Overlapped bboxes with IoU higher than this
        threshold will be filtered.
    :return: Remaining bboxes and its scores.
    """
    valid_idx = bboxes[:, 0] < bboxes[:, 1]
    scores = scores[valid_idx]
    bboxes = bboxes[valid_idx]

    arg_desc = scores.argsort()[::-1]

    scores_remain = scores[arg_desc]
    bboxes_remain = bboxes[arg_desc]

    keep_bboxes = []
    keep_scores = []

    while bboxes_remain.size > 0:
        bbox = bboxes_remain[0]
        score = scores_remain[0]
        keep_bboxes.append(bbox)
        keep_scores.append(score)

        iou = iou_lr(bboxes_remain, np.expand_dims(bbox, axis=0))

        keep_indices = (iou < thresh)
        bboxes_remain = bboxes_remain[keep_indices]
        scores_remain = scores_remain[keep_indices]

    keep_bboxes = np.asarray(keep_bboxes, dtype=bboxes.dtype)
    keep_scores = np.asarray(keep_scores, dtype=scores.dtype)

    return keep_scores, keep_bboxes

def get_loc_label(target: np.ndarray) -> np.ndarray:
    """Generate location offset label from ground truth summary.

    :param target: Ground truth summary. Sized [N].
    :return: Location offset label in LR format. Sized [N, 2].
    """
    seq_len, = target.shape

    bboxes = seq2bbox(target)
    offsets = bbox2offset(bboxes, seq_len)

    return offsets


def get_ctr_label(target: np.ndarray,
                  offset: np.ndarray,
                  eps: float = 1e-8
                  ) -> np.ndarray:
    """Generate centerness label for ground truth summary.

    :param target: Ground truth summary. Sized [N].
    :param offset: LR offset corresponding to target. Sized [N, 2].
    :param eps: Small floating value to prevent division by zero.
    :return: Centerness label. Sized [N].
    """
    target = np.asarray(target, dtype=bool)
    ctr_label = np.zeros(target.shape, dtype=np.float32)

    offset_left, offset_right = offset[target, 0], offset[target, 1]
    ctr_label[target] = np.minimum(offset_left, offset_right) / (
        np.maximum(offset_left, offset_right) + eps)

    return ctr_label


def bbox2offset(bboxes: np.ndarray, seq_len: int) -> np.ndarray:
    """Convert LR bounding boxes to LR offsets.

    :param bboxes: LR bounding boxes.
    :param seq_len: Sequence length N.
    :return: LR offsets. Sized [N, 2].
    """
    pos_idx = np.arange(seq_len, dtype=np.float32)
    offsets = np.zeros((seq_len, 2), dtype=np.float32)

    for lo, hi in bboxes:
        bbox_pos = pos_idx[lo:hi]
        offsets[lo:hi] = np.vstack((bbox_pos - lo, hi - 1 - bbox_pos)).T

    return offsets


def offset2bbox(offsets: np.ndarray) -> np.ndarray:
    """Convert LR offsets to bounding boxes.

    :param offsets: LR offsets. Sized [B, N, 2].
    :return: Bounding boxes corresponding to offsets. Sized [B, N, 2].
    """
    offset_left, offset_right = offsets[:, :, 0], offsets[:, :, 1] #[B, N]
    B, seq_len, _ = offsets.shape
    indices = np.arange(seq_len).reshape(1, seq_len)
    indices = np.repeat(indices, B, axis=0) #[B, N]
    bbox_left = indices - offset_left #[B, N]
    bbox_right = indices + offset_right + 1 #[B, N]
    bboxes = np.stack((bbox_left, bbox_right), axis=-1) #[B, N, 2]
    return bboxes
