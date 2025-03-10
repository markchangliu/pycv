import os
from typing import Union, List, Tuple, Literal

import copy
import numpy as np
import pycocotools.mask as pycocomask


def assign_gt_to_dt(
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
    - `dt_gt_ids`: `Array[long]`, shape `(num_pred, )`,
    和dt匹配的最大分数的gt索引
    - `dt_tp_flags`: `Array[bool]`, shape `(num_pred, )`, 
    - `gt_fn_flags`: `Array[bool]`, shape `(num_gt, )`,

    Examples
    ```
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
    gt_fn_flags = np.zeros((num_gt, ), dtype=np.bool_)

    if num_pred == 0 or np.max(score_mat) < thres or num_gt == 0:
        return dt_gt_ids, dt_tp_flags, gt_labels

    # 为dt匹配分数最大的gt
    dt_gt_ids = np.argmax(score_mat, axis=1)
    max_scores_dt = score_mat[range(num_pred), dt_gt_ids]

    # 将分数大于thres的pred label设为True
    dt_tp_flags[max_scores_dt>thres] = True

    # 将匹配到dt的gt label设为True
    max_scores_gt = np.max(score_mat, axis=0)
    gt_labels = max_scores_gt > thres

    # 如果max_one_dt_per_gt为True, 将除max_score以外的dt_labels改为False
    if max_one_dt_per_gt:
        dt_tp_flags[max_scores_dt<max_scores_gt[dt_gt_ids]] = False
    
    gt_fn_flags = ~gt_labels
    
    return dt_gt_ids, dt_tp_flags, gt_fn_flags


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
) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray]:
    """
    确定每个dt_insts是否为fp, 每个gt_insts是否为fn, 并计算AP.

    Args
    - `gt_insts`: bbox或者segm
        - bbox: `Array[int]`, shape `(num_gt, 4)`, xywh
        - segm: `Array[uint8]`, shape `(num_gt, max_h, max_w)`, 0/1, 
        pad至最大图片尺寸, 需要用`gt_img_hws`还原
    - `gt_img_ids`: `Array[int]`, shape `(num_gt, )`
    - `gt_img_hws`: `Array[int]`, shape `(num_gt, 2)`, 原图大小, 用于还原mask
    - `dt_insts`: bbox或者segm
        - bbox: `Array[int]`, shape `(num_dt, 4)`, xywh
        - segm: `Array[uint8]`, shape `(num_dt, max_h, max_w)`, 0/1, 
        pad至最大图片尺寸, 需要用`dt_img_hws`还原
    - `dt_img_ids`: `Array[int]`, shape `(num_dt, )`
    - `dt_img_hws`: `Array[int]`, shape `(num_dt, 2)`, 原图大小, 用于还原mask
    - `dt_scores`: `Array[float]`, shape `(num_dt, )`
    - `iou_thres`: `float`

    Returns
    - `metrics`: `Tuple[float, float, float]`, `[AP, prec, rec]`
    - `dt_gt_ids`: `Array[uint]`, 匹配的gt id
    - `dt_tp_flags`: `Array[bool]`, shape `(num_dt, )`
    - `gt_fn_flags`: `Array[bool]`, shape `(num_gt, )`,
    """
    dt_gt_ids = np.empty(len(dt_insts), dtype=np.uint8)
    dt_tp_flags = np.empty(len(dt_insts), dtype=np.uint8)
    gt_fn_flags = np.empty(len(gt_insts), dtype=np.uint8)
    img_ids = set(gt_img_ids) | set(dt_img_ids)

    if len(gt_insts.shape) == 2:
        mode = "bbox"
    elif len(gt_insts) == 3:
        mode = "segm"

    # 遍历图片, 在每张图片上匹配dt和gt, 标记dt是否为tp, gt是否为fn
    for img_id in img_ids:
        gt_indice_imgi = gt_img_ids == img_id
        gt_insts_imgi = gt_insts[gt_indice_imgi]

        dt_indice_imgi = dt_img_ids == img_id
        dt_insts_imgi = dt_insts[dt_indice_imgi]

        img_h, img_w = gt_img_hws[dt_indice_imgi][0].tolist()

        if mode == "bbox":
            iou = get_iou_bbox(gt_insts_imgi, dt_insts_imgi)
        else:
            iou = get_iou_segm(gt_insts_imgi, dt_insts_imgi)
        
        dt_gt_ids_imgi, dt_tp_flags_imgi, gt_fn_flags_imgi = assign_gt_to_dt(
            iou, iou_thres, True
        )

        dt_gt_ids[dt_indice_imgi] = dt_gt_ids_imgi
        dt_tp_flags[dt_indice_imgi] = dt_tp_flags_imgi
        gt_fn_flags[gt_indice_imgi] = gt_fn_flags_imgi
    
    # 将dt按照confidence排序, 创建prec-rec curve, 计算AP
    dt_sort_indice = np.argsort(dt_scores)[::-1]
    dt_tp_flags_sort = dt_tp_flags[dt_sort_indice]

    # 用 culmulative sum 计算prec_curve和rec_curve
    # i.e. [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
    num_dts = len(dt_insts)
    num_gts = len(gt_insts)
    ep = 1e-6
    tp_cumsum = np.cumsum(dt_tp_flags_sort.astype(np.uint))
    prec_curve = tp_cumsum / (num_dts + ep)
    rec_curve = tp_cumsum / (num_gts + ep)

    # 用积分计算prec_curve和rec_curve下的面积, 即AP
    # 注意初始点坐标为(prec=1, rec=0)
    prec_curve = np.concatenate([1], prec_curve)
    rec_curve = np.concatenate([0], rec_curve)
    ap = np.trapz(prec_curve, rec_curve)
    prec = prec_curve[-1].item()
    rec = rec_curve[-1].item()
    metrics = (ap, prec, rec)

    return metrics, dt_gt_ids, dt_tp_flags, gt_fn_flags