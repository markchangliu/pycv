import os
from typing import Union, List, Tuple

import copy
import numpy as np
import pycocotools.mask as pycocomask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def assign_gt_to_dt(
    score_mat: np.ndarray,
    thres: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为dt匹配gt，参考：
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/matcher.py
    

    Args
    - `score_mat`: `Array[float]`, shape `(num_pred, num_gt)`
    - `thres`: `float`

    Returns
    - `dt_gt_ids`: `Array[long]`, shape `(num_pred, )`,
    和dt匹配的最大分数的gt索引
    - `dt_labels`: `Array[bool]`, shape `(num_pred, )`, 
    dt最大分数是否超过`thres`
    - `gt_labels`: `Array[bool]`, shape `(num_gt, )`,
    gt是否有dt匹配
    """
    num_pred, num_gt = score_mat.shape
    score_mat = copy.deepcopy(score_mat)

    dt_gt_ids = np.ones((num_pred, ), dtype=np.long) * -1
    dt_labels = np.zeros((num_pred, ), dtype=np.bool_)
    gt_labels = np.zeros((num_gt, ), dtype=np.bool_)

    if num_pred == 0 or np.max(score_mat) < thres or num_gt == 0:
        return dt_gt_ids, dt_labels, gt_labels

    # 为dt匹配分数最大的gt
    max_scores_gt_ids = np.argmax(score_mat, axis=1)
    max_scores_dt = score_mat[range(num_pred), max_scores_gt_ids]

    # 将分数大于thres的pred label设为True
    dt_gt_ids = max_scores_gt_ids
    dt_labels[max_scores_dt>thres] = True

    # 将匹配到dt的gt label设为True
    max_scores_dt_ids = np.argmax(score_mat, axis=0)
    max_scores_gt = score_mat[, ]
    
    return dt_gt_ids, dt_labels


def get_iou_bbox(
    gt: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Args
    - `gt`: `Array[int]`, shape `(num_gt, 4)`, `x1y1wh`
    - `dt`: `Array[int]`, shape `(num_dt, 4)`, `x1x2wh`

    Returns
    - `iou_mat`: `Array[float]`, shape `(num_dt, num_gt)`
    """
    iou = pycocomask.iou(dt, gt, [False] * len(dt))
    return iou


def get_tp_fn(
    gt: np.ndarray,
    dt: np.ndarray,
    iou_thres: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据iou判断dt为TP，gt是否为FN。dt和gt为同一类别。

    Args
    - `gt`: `Array[int]`, shape `(num_gt, 4)`, `x1y1wh`
    - `dt`: `Array[int]`, shape `(num_dt, 4)`, `x1y1wh`
    - `iou_thres`: `float`

    Returns
    - `tp`: `Array[bool]`, shape `(num_dt, )`
    - `fn`: `Array[bool]`, shape `(num_gt, )`
    """
    iou_mat = get_iou_bbox(gt, dt)
    dt_gt_ids, dt_labels = assign_gt_to_dt(iou_mat, iou_thres)

    tp = np.empty(dt.shape, np.bool_)
    fn = np.empty(gt.shape, np.bool_)




def get_mAP_bbox(
    gt_p: Union[str, os.PathLike],
    dt_p: Union[str, os.PathLike],
    iou_thres: List[float],
    tags: List[str],
    categories: List[int]
) -> None:
    coco_gt = COCO(gt_p)
    coco_dt = coco_gt.loadRes(dt_p)

