import copy
import numpy as np


def bboxes_xywh2xyxy(
    bboxes_xywh: np.ndarray
) -> np.ndarray:
    """
    Args
    - `bboxes_xywh`: `Array[int]`, shape `(num_bbox, 4)`

    Returns
    - `bboxes_xyxy`: `Array[int]`, shape `(num_bbox, 4)`
    """
    ws = bboxes_xywh[:, 2]
    hs = bboxes_xywh[:, 3]
    
    bboxes_xyxy = copy.deepcopy(bboxes_xywh)
    bboxes_xyxy[:, 2] = bboxes_xywh[:, 0] + ws
    bboxes_xyxy[:, 3] = bboxes_xywh[:, 1] + hs

    return bboxes_xyxy


def bboxes_xyxy2xywh(
    bboxes_xyxy: np.ndarray
) -> np.ndarray:
    """
    Args
    - `bboxes_xyxy`: `Array[int]`, shape `(num_bbox, 4)`

    Returns
    - `bboxes_xywh`: `Array[int]`, shape `(num_bbox, 4)`
    """
    ws = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
    hs = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]

    bboxes_xywh = copy.deepcopy(bboxes_xyxy)
    bboxes_xywh[:, 2] = ws
    bboxes_xywh[:, 3] = hs

    return bboxes_xywh
