import os
from typing import Union, List, Tuple, Literal

import copy
import numpy as np
import pycocotools.mask as pycocomask

from pycv.labels.coco import (
    load_coco_dt, load_coco_gt, get_anns_of_img, get_dts_of_img
)


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


def get_mAP_prec_rec(
    gt_p: Union[str, os.PathLike],
    dt_p: Union[str, os.PathLike],
    iou_thres: List[float],
    category_ids: List[int],
    ann_tags: List[str],
    mode: Literal["bbox", "segm"],
    export_gt_eval_res_p: Union[str, os.PathLike],
    export_dt_eval_res_p: Union[str, os.PathLike],
) -> None:
    imgs, categories, gts = load_coco_gt(
        gt_p, category_ids, ann_tags
    )
    dts = load_coco_dt(dt_p, category_ids)

    # 将dts按照score从大到小排序
    dts: List[dict] = list(dts.values())
    dts.sort(key=lambda e: e["score"], reverse = True)

    for thres in iou_thres:
        # 记录每个dt是否为tp，gt是否有dt匹配
        dt_labels = np.empty((len(dts), ), dtype=np.uint8)
        gt_labels = np.zeros(len(gts.values()), dtype=np.uint8)

        for dt in dts:
            img_id = dt["image_id"]
            gts_of_img = get_anns_of_img(gts, img_id)

            if mode == "bbox":
                dt_bbox = np.asarray(dt["bbox"]).reshape(-1, 4)
                gt_bboxes = [gt["bbox"] for gt in gts_of_img]
                gt_bboxes = np.asarray(gt_bboxes).reshape(-1, 4)
                iou = get_iou_bbox(gt_bboxes, dt_bbox)
            elif mode == "segm":
                dt_