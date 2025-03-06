import os
from typing import Union, List, Tuple, Literal

import copy
import numpy as np
import pycocotools.mask as pycocomask


def assign_gt_to_dt(
    score_mat: np.ndarray,
    thres: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    max_scores_gt = np.max(score_mat, axis=0)
    gt_labels = max_scores_gt > thres
    
    return dt_gt_ids, dt_labels, gt_labels


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
    iou = pycocomask.iou(dt, gt, [False] * len(gt))
    return iou


def get_iou_segm(
    gt: np.ndarray,
    dt: np.ndarray
) -> np.ndarray:
    """
    Args
    - `gt`: `Array[uint8]`, 0/1, shape `(num_gt, img_h, img_w)`
    - `dt`: `Array[uint8]`, 0/1, shape `(num_dt, img_h, img_w)`

    Returns
    - `iou`: `Array[float]`, shape `(num_dt, num_gt)`
    """
    gt = pycocomask.encode(gt)
    dt = pycocomask.encode(dt)
    iou = pycocomask.iou(dt, gt, [False] * len(gt))
    return iou


def get_AP_fp_fn(
    gt_insts: np.ndarray,
    gt_img_ids: np.ndarray,
    gt_img_hws: np.ndarray,
    dt_insts: np.ndarray,
    dt_img_ids: np.ndarray,
    dt_img_hws: np.ndarray,
    dt_scores: np.ndarray,
    iou_thres: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    确定每个dt_insts是否为fp, 每个gt_insts是否为fn, 并计算AP.

    Args
    - `gt_insts`: bbox或者segm
        - bbox: `Array[int]`, shape `(num_gt_insts, 4)`, xywh
        - segm: `Array[uint8]`, shape `(num_gt_insts, max_h, max_w)`, 0/1, 
        pad至最大图片尺寸, 需要用`gt_img_hws`还原
    - `gt_img_ids`: `Array[int]`, shape `(num_gt_insts, )`
    - `gt_img_hws`: `Array[int]`, shape `(num_gt_insts, 2)`, 原图大小, 用于还原mask
    - `dt_insts`: bbox或者segm
        - bbox: `Array[int]`, shape `(num_dt_insts, 4)`, xywh
        - segm: `Array[uint8]`, shape `(num_dt_insts, max_h, max_w)`, 0/1, 
        pad至最大图片尺寸, 需要用`dt_img_hws`还原
    - `dt_img_ids`: `Array[int]`, shape `(num_dt_insts, )`
    - `dt_img_hws`: `Array[int]`, shape `(num_dt_insts, 2)`, 原图大小, 用于还原mask
    - `dt_scores`: `Array[float]`, shape `(num_dt_insts, )`
    - `iou_thres`: `float`

    Returns
    - `AP`: `float`
    - `dt_gt_ids`: `Array[uint]`, 匹配的gt idx
    - `dt_labels`: `Array[bool]`, shape `(num_dt_insts, )`, 是否为tp
    - `gt_labels`: `Array[bool]`, shape `(num_gt_insts, )`, 是否为fn
    """
    dt_gt_ids = np.empty(len(dt_insts), dtype=np.uint8)
    dt_labels = np.empty(len(dt_insts), dtype=np.uint8)
    gt_labels = np.empty(len(gt_insts), dtype=np.uint8)
    img_ids = set(gt_img_ids) | set(dt_img_ids)

    if len(gt_insts.shape) == 2:
        mode = "bbox"
    elif len(gt_insts) == 3:
        mode = "segm"

    for img_id in img_ids:
        gt_indice = gt_img_ids == img_id
        gt_insts_imgi = gt_insts[gt_indice]

        dt_indice = dt_img_ids == img_id
        dt_scores_imgi = dt_scores[dt_indice]
        dt_insts_imgi = dt_insts[dt_indice]

        img_h, img_w = gt_img_hws[gt_indice][0].tolist()

        if mode == "bbox":
            iou = get_iou_bbox(gt_insts_imgi, dt_insts_imgi)
        else:
            iou = get_iou_segm(gt_insts_imgi, dt_insts_imgi)
        
        dt_gt_ids_imgi, dt_labels_imgi, gt_labels_imgi = assign_gt_to_dt(
            iou, iou_thres
        )

        dt_gt_ids[dt_indice] = dt_gt_ids_imgi
        