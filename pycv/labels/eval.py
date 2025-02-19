import copy
import json
import os
from typing import List, Union, Literal

import numpy as np
import pycocotools.mask as pycocomask
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def eval_coco_mAP(
    coco_gt: Union[str, os.PathLike, COCO],
    coco_pred_res_p: Union[str, os.PathLike],
    iou_thres_list: List[float],
    eval_mode: Literal["bbox", "segm"],
) -> None:
    if isinstance(coco_gt, str):
        coco_gt = COCO(coco_gt)
    
    coco_pred = coco_gt.loadRes(coco_pred_res_p)
    
    img_ids = coco_gt.getImgIds()

    coco_eval = COCOeval(coco_gt, coco_pred, eval_mode)
    coco_eval.params.imgIds = img_ids
    coco_eval.params.iouThrs = np.array(iou_thres_list)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def eval_coco_prec_rec(
    coco_gt: Union[str, os.PathLike, COCO],
    coco_pred_res_p: Union[str, os.PathLike],
    iou_thres_list: List[float],
    eval_mode: Literal["bbox", "segm"]
) -> None:
    if isinstance(coco_gt, str):
        coco_gt = COCO(coco_gt)
    
    coco_pred = coco_gt.loadRes(coco_pred_res_p)
    img_ids = coco_gt.getImgIds()

    eval_res = {
        "num_preds": [],
        "num_tps": [],
        "num_fps": [],
        "num_fns": [],
        "precision": [],
        "recall": [],
        "avg_tp_iou": []
    }

    for iou_thres in iou_thres_list:
        num_preds = 0
        num_tp = 0
        num_fp = 0
        num_fn = 0
        sum_tp_iou = 0

        for img_id in img_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            img_h = img_info["height"]
            img_w = img_info["width"]

            gt_ann_ids = coco_gt.getAnnIds(img_id)
            gt_anns = coco_gt.loadAnns(gt_ann_ids)
            pred_ann_ids = coco_pred.getAnnIds(img_id)
            pred_anns = coco_pred.loadAnns(pred_ann_ids)

            if len(pred_anns) == 0:
                num_fn += len(gt_anns)
                continue
            elif len(gt_anns) == 0:
                num_fp += len(pred_anns)
                num_preds += len(pred_anns)
                continue

            # compute iou
            if eval_mode == "bbox":
                # [[x1, y1, w, h], ...]
                pred_objs = [pred_ann["bbox"] for pred_ann in pred_anns]
                gt_objs = [gt_ann["bbox"] for gt_ann in gt_anns]
                iscrowd = [0] * len(gt_objs)
            elif eval_mode == "segm":
                gt_polys = [gt_ann["segmentation"] for gt_ann in gt_anns]
                gt_objs = [pycocomask.frPyObjects(gt_poly, img_h, img_w) for gt_poly in gt_polys]
                gt_objs = [pycocomask.merge(rles) for rles in gt_objs]
                pred_objs = [pred_ann["segmentation"] for pred_ann in pred_anns]
                iscrowd = [0] * len(gt_objs)

            # (num_preds, num_gts)
            ious = pycocomask.iou(pred_objs, gt_objs, iscrowd)
            
            # assign pred to gt
            max_ious_pred = np.max(ious, axis=1) # (num_preds, )
            max_ious_gt = np.max(ious, axis=0) # (num_gts, )

            # compute tp, fp, fn
            num_preds += len(max_ious_pred)
            num_tp += np.sum(max_ious_pred >= iou_thres)
            num_fp += np.sum(max_ious_pred < iou_thres)
            num_fn += np.sum(max_ious_gt < iou_thres)
            sum_tp_iou += np.sum(max_ious_pred[max_ious_pred >= iou_thres])
        
        precision = num_tp / (num_tp + num_fp)
        recall = num_tp / (num_tp + num_fn)
        avg_tp_iou = sum_tp_iou / num_tp

        eval_res["num_preds"].append(num_preds)
        eval_res["num_tps"].append(num_tp)
        eval_res["num_fps"].append(num_fp)
        eval_res["num_fns"].append(num_fn)
        eval_res["precision"].append(round(precision, 3))
        eval_res["recall"].append(round(recall, 3))
        eval_res["avg_tp_iou"].append(round(avg_tp_iou, 3))

    # compute overall res
    overall_res = {
        "avg_precision": sum(eval_res["precision"]) / len(eval_res["precision"]),
        "avg_recall": sum(eval_res["recall"]) / len(eval_res["recall"])
    }

    # print res
    print()
    print("num_preds:", eval_res["num_preds"])
    print("num_tps:", eval_res["num_tps"])
    print("num_fps:", eval_res["num_fps"])
    print("num_fns:", eval_res["num_fns"])
    print("precision:", eval_res["precision"])
    print("recall:", eval_res["recall"])
    print("avg_tp_iou:", eval_res["avg_tp_iou"])
    print("avg_precision:", overall_res["avg_precision"])
    print("avg_recall:", overall_res["avg_recall"])
    print()