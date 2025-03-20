import copy
from typing import Tuple, Literal, Union

import numpy as np
import pycocotools.mask as pycocomask

from pycv.data_structures.insts import Insts


class BaseMatcher:
    def match(
        self, 
        dts: Insts, 
        gts: Insts
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args
        - `dts`: `Insts`,
        - `gts`: `Insts`

        Returns
        - `dt_gt_ids`: `Array[int]`, shape `(num_dts, )`
        - `tp_flags`: `Array[bool]`, shape `(num_dts, )`
        - `fn_flags`: `Array[bool]`, shape `(num_gts, )`
        """
        raise NotImplementedError


class IoUMatcher(BaseMatcher):
    def __init__(
        self,
        iou_thres: float,
        max_one_dt_per_gt: bool,
        mode: Literal["bbox", "segm"]
    ) -> None:
        self.iou_thres = iou_thres
        self.max_one_dt_per_gt = max_one_dt_per_gt
        self.mode = mode
    
    def match(
        self, 
        dts: Insts,
        gts: Insts
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        iou_mat = get_iou(gts, dts, self.mode)
        dt_gt_ids, tp_flags, fn_flags = match_by_score_mat(
            iou_mat, self.iou_thres, self.max_one_dt_per_gt
        )
        return dt_gt_ids, tp_flags, fn_flags


def match_by_score_mat(
    score_mat: np.ndarray,
    thres: float,
    max_one_dt_per_gt: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为dt匹配gt，参考：
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/matcher.py
    

    Args
    - `score_mat`: `Array[float]`, shape `(num_pred, num_gt)`
    - `thres`: `float`
    - `max_one_dt_per_gt`: `bool`
        - `True`: 一个gt最多匹配一个dt
        - `False`: 一个gt可以匹配无数个dt

    Returns
    - `dt_gt_ids`: `Array[int]`, shape `(num_pred, )`,
    和dt匹配的最大分数的gt索引
    - `dt_tp_flags`: `Array[bool]`, shape `(num_pred, )`, 
    - `gt_fn_flags`: `Array[bool]`, shape `(num_gt, )`,

    Examples
    ```python
    score_mat = [
        [0.4, 0.1, 0.3],
        [0.3, 0.8, 0.2],
        [0.1, 0.9, 0.4],
        [0.1, 0.2, 0.7]
    ]

    # max_one_dt_per_gt为False, gt 1匹配了dt 1和dt 2
    dt_gt_ids, dt_tp_flags, gt_fn_flags = assign_gt_to_dt(
        score_mat, 0.5, False
    )

    print(dt_gt_ids)
    >>> array([0, 1, 1, 2], dtype=int64)

    print(dt_tp_flags)
    >>> array([False,  True,  True,  True])

    print(gt_fn_flags)
    >>> array([True,  False,  False])

    # max_one_dt_per_gt为True, gt 1只匹配了dt 1
    dt_gt_ids, dt_tp_flags, gt_fn_flags = assign_gt_to_dt(
        score_mat, 0.5, True
    )

    print(dt_gt_ids)
    >>> array([0, 1, 1, 2], dtype=int64)

    print(dt_tp_flags)
    >>> array([False,  False,  True,  True])

    print(gt_fn_flags)
    >>> array([True,  False,  False])
    ```
    """
    num_pred, num_gt = score_mat.shape
    score_mat = copy.deepcopy(score_mat)

    dt_gt_ids = np.ones((num_pred, ), dtype=np.int_) * -1
    dt_tp_flags = np.zeros((num_pred, ), dtype=np.bool_)
    gt_matched_flags = np.zeros((num_gt, ), dtype=np.bool_)

    if num_pred == 0 or np.max(score_mat) < thres or num_gt == 0:
        gt_fn_flags = ~gt_matched_flags
        return dt_gt_ids, dt_tp_flags, gt_fn_flags

    # 为dt匹配分数最大的gt
    dt_gt_ids = np.argmax(score_mat, axis=1)
    max_scores_dt = score_mat[range(num_pred), dt_gt_ids]

    # 将分数大于thres的pred label设为True
    dt_tp_flags[max_scores_dt>thres] = True

    # 将匹配到dt的gt label设为True
    max_scores_gt = np.max(score_mat, axis=0)
    gt_matched_flags = max_scores_gt > thres

    # 如果max_one_dt_per_gt为True, 将除max_score以外的dt_labels改为False
    if max_one_dt_per_gt:
        dt_tp_flags[max_scores_dt<max_scores_gt[dt_gt_ids]] = False
    
    gt_fn_flags = ~gt_matched_flags
    
    return dt_gt_ids, dt_tp_flags, gt_fn_flags
        

def get_iou(
    gts: np.ndarray,
    dts: np.ndarray,
    mode: Literal["bbox", "segm"]
) -> np.ndarray:
    if mode == "bbox":
        iou_mat = get_iou_bboxes(gts, dts)
    elif mode == "segm":
        iou_mat = get_iou_masks(gts, dts)
    else:
        raise NotImplementedError
    
    return iou_mat


def get_iou_bboxes(
    gts: np.ndarray,
    dts: np.ndarray,
) -> np.ndarray:
    """
    Args
    - `gts`: `Array[int]`, shape `(num_gt, 4)`, `x1y1wh`
    - `dts`: `Array[int]`, shape `(num_dt, 4)`, `x1x2wh`

    Returns
    - `iou_mat`: `Array[float]`, shape `(num_dt, num_gt)`
    """
    iou_mat = pycocomask.iou(dts, gts, [False] * len(gts))
    return iou_mat


def get_iou_masks(
    gts: np.ndarray,
    dts: np.ndarray
) -> np.ndarray:
    """
    Args
    - `gts`: `Array[uint8]`, 0/1, shape `(num_gt, img_h, img_w)`
    - `dts`: `Array[uint8]`, 0/1, shape `(num_dt, img_h, img_w)`

    Returns
    - `iou_mat`: `Array[float]`, shape `(num_dt, num_gt)`
    """
    gts = pycocomask.encode(gts)
    dts = pycocomask.encode(dts)
    iou_mat = pycocomask.iou(dts, gts, [False] * len(gts))
    return iou_mat