import os
import json
import shutil
from pathlib import Path
from typing import Union, Literal, List, Tuple, Dict

import numpy as np
import pycocotools.mask as pycocomask
from pycocotools.coco import COCO

from pycv.structures_bkp import DetInsts, SegmInsts
from pycv.labels.insts import insts2labelme
from pycv.data_structures.masks import polyscoco2masks, rles2masks, center_pad_masks


def coco2labelme(
    ann_p: Union[str, os.PathLike],
    img_prefix: Union[str, os.PathLike],
    with_mask: bool
) -> None:
    coco = COCO(ann_p)

    cat_id_name_dict = {}
    for i, cat_info in coco.cats.items():
        cat_id = cat_info["id"]
        cat_name = cat_info["name"]
        cat_id_name_dict[cat_id] = cat_name

    for i, img_info in coco.imgs.items():
        img_p = os.path.join(img_prefix, img_info["file_name"])
        img_w = img_info["width"]
        img_h = img_info["height"]
        img_id = img_info["id"]

        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)
        bboxes = []
        cat_ids = []
        
        for ann in anns:
            cat_id = ann["category_id"]
            cat_ids.append(cat_id)
            cat_name = cat_id_name_dict[cat_id]

            if not with_mask:
                x1, y1, w, h = ann["bbox"]
                x2, y2 = x1 + w, y1 + h
                bbox = [x1, y1, x2, y2]
                bbox = [int(i) for i in bbox]
                bboxes.append(bbox)
            else:
                raise NotImplementedError

        bboxes = np.asarray(bboxes)
        cat_ids = np.asarray(cat_ids)
        scores = np.array([1] * len(cat_ids))
        insts = DetInsts(scores, cat_ids, bboxes)

        img_p = Path(img_p)
        img_name = img_p.name
        img_stem = img_p.stem
        img_folder = img_p.parent
        export_json_p = os.path.join(img_folder, f"{img_stem}.json")

        insts2labelme(
            insts, img_name, export_json_p, (img_h, img_w),
            cat_id_name_dict
        )


def coco2yolo(
    coco_p: Union[str, os.PathLike],
    export_root: Union[str, os.PathLike],
    export_subdir: Union[str, os.PathLike],
    coco_img_prefix: str = "",
    ann_mode: Literal["det", "seg"] = "seg"
) -> None:
    coco = COCO(coco_p)

    yolo_img_dir = os.path.join(export_root, "images", export_subdir)
    yolo_label_dir = os.path.join(export_root, "labels", export_subdir)

    os.makedirs(yolo_img_dir, exist_ok=True)
    os.makedirs(yolo_label_dir, exist_ok=True)

    for i, img_info in coco.imgs.items():
        img_id = img_info["id"]
        img_h = img_info["height"]
        img_w = img_info["width"]

        if len(coco_img_prefix) > 0:
            img_p = os.path.join(coco_img_prefix, img_info["file_name"])
        else:
            img_p = img_info["file_name"]
        img_suffix = Path(img_p).suffix

        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)

        yolo_anns = []

        # normalize polygons
        for ann in anns:
            cat_id = ann["category_id"]

            if ann_mode == "seg":
                polys = ann["segmentation"]
                polys = [np.asarray(poly, dtype = np.float32).reshape(-1, 2) for poly in polys]
                polys = np.concatenate(polys, axis=0)
                polys[:, 0] = polys[:, 0] / img_w
                polys[:, 1] = polys[:, 1] / img_h
                polys = polys.round(3)
                yolo_anns.append((cat_id, polys))
            elif ann_mode == "det":
                bbox = np.asarray(ann["bbox"]).astype(np.float32)
                bbox[:2] += bbox[2:] / 2  # xy top-left corner to center
                bbox[[0, 2]] /= img_w  # normalize x
                bbox[[1, 3]] /= img_h  # normalize y
                yolo_anns.append((cat_id, bbox))

        # save img
        yolo_img_p = os.path.join(yolo_img_dir, f"{img_id}{img_suffix}")
        shutil.copy(img_p, yolo_img_p)

        # save labels
        yolo_ann_p = os.path.join(yolo_label_dir, f"{img_id}.txt")

        with open(yolo_ann_p, "w") as f:
            if ann_mode == "seg":
                for cat_id, polys in yolo_anns:
                    polys: np.ndarray
                    polys = polys.flatten().tolist()
                    polys = [str(p) for p in polys]
                    polys_txt = " ".join(polys)
                    label_txt = f"{cat_id} {polys_txt}\n"
                    f.write(label_txt)
            elif ann_mode == "det":
                for cat_id, bbox in yolo_anns:
                    bbox: np.ndarray
                    bbox = bbox.flatten().tolist()
                    bbox = [str(b) for b in bbox]
                    bbox_txt = " ".join(bbox)
                    label_txt = f"{cat_id} {bbox_txt}\n"
                    f.write(label_txt)


def load_coco_gt(
    coco_gt_p: Union[str, os.PathLike],
    category_ids: Union[Literal["all"], List[str]],
    ann_tags: Union[Literal["all"], List[str]]
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Args
    - `coco_gt_p`: `Union[str, os.PathLike]`
    - `category_ids`: `Union[Literal["all"], List[int]]`, 要包含的category id, 
    `"all"`表示所有
    - `ann_tags`: `Union[Literal["all"], List[str]]`, 要包含的ann tag, 
    `"all"`表示所有

    Returns
    - `img_infos`: `List[dict]`, info keys `height`, `width`, `id`, `file_name`
    - `category_infos`: `List[dict]`, info keys `id`, `name`
    - `ann_infos`: `List[dict]`, keys `id`, `iscrowd`, `image_id`, `area`, `bbox`, 
    `segmentation`, `category_id`, `ann_tags`
    """
    with open(coco_gt_p, "r") as f:
        coco_gt = json.load(f)

    img_infos: List[dict] = coco_gt["images"]
    category_infos: List[dict] = coco_gt["categories"]
    ann_infos: List[dict] = coco_gt["annotations"]

    # 过滤掉不要的category_id
    if isinstance(category_ids, list):
        category_infos = [c for c in category_infos if c["id"] in category_ids]
    elif isinstance(category_ids, str) and category_ids == "all":
        category_ids = [c["id"] for c in category_infos]
    
    ann_infos = [a for a in ann_infos if a["category_id"] in category_ids]

    if isinstance(ann_tags, str) and ann_tags == "all":
        return img_infos, category_infos, ann_infos
    
    # 过滤掉不要的tag
    ann_infos_ = []
    for ann_info in ann_infos:
        ann_tags = ann_info["tags"]
        exclude_flag = True

        for t in ann_tags:
            if t in ann_tags:
                exclude_flag = False
                break
        
        if not exclude_flag:
            ann_infos_.append(ann_info)
    
    ann_infos = ann_infos_
    del ann_infos_

    return img_infos, category_infos, ann_infos


def load_coco_dt(
    coco_dt_p: Union[str, os.PathLike],
    category_ids: Union[Literal["all"], List[str]],
) -> List[dict]:
    """
    Args
    - `coco_dt_p`: `Union[str, os.PathLike]`
    - `category_ids`: `Union[Literal["all"], List[int]]`, 要包含的category id, 
    `"all"`表示所有

    Returns
    - `dt_infos`: `List[dict]`, info keys `image_id`, `category_id`, `bbox`, 
    `segmentation`, `score`
    """
    with open(coco_dt_p, "r") as f:
        coco_dt: List[dict] = json.load(f)
    
    if isinstance(category_ids, str) and category_ids == "all":
        return coco_dt
    
    # 过滤掉不要的category_id
    coco_dt = [d for d in coco_dt if d["category_id"] in category_ids]

    return coco_dt


def get_anns_of_img(
    ann_infos: List[dict],
    img_id: int,
) -> List[dict]:
    """
    Args
    - `ann_infos`: `List[dict]`, info keys `id`, `iscrowd`, `image_id`, 
    `area`, `bbox`, `segmentation`, `category_id`, `ann_tags`
    - `img_id`: int

    Returns
    - `anns_of_img`: `List[dict]`
    """
    anns_of_img = [a for a in ann_infos if a["image_id"] == img_id]

    return anns_of_img

def get_anns_of_category(
    ann_infos: List[dict],
    category_id: int
) -> List[dict]:
    """
    Args
    - `ann_infos`: `List[dict]`, info keys `id`, `iscrowd`, `image_id`, 
    `area`, `bbox`, `segmentation`, `category_id`, `ann_tags`
    - `category_id`: int

    Returns
    - `anns_of_category`: `List[dict]`
    """
    anns_of_category = [a for a in ann_infos if a["category_id"] == category_id]
    return anns_of_category


def get_dts_of_img(
    dt_infos: List[dict],
    img_id: int,
) -> List[dict]:
    """
    Args
    - `dt_infos`: `List[dict]`, info keys `image_id`, `category_id`, 
    `bbox`, `segmentation`, `score`
    - `img_id`: int

    Returns
    - `dts_of_img`: `List[dict]`
    """
    dts_of_img = [v for v in dt_infos if v["image_id"] == img_id]

    return dts_of_img


def get_dts_of_category(
    dt_infos: List[dict],
    category_id: int,
) -> List[dict]:
    """
    Args
    - `dt_infos`: `List[dict]`, info keys `image_id`, `category_id`, 
    `bbox`, `segmentation`, `score`
    - `category_id`: int

    Returns
    - `dts_of_category`: `List[dict]`
    """
    dts_of_category = [v for v in dt_infos if v["image_id"] == category_id]
    return dts_of_category


def get_mAP_prec_rec(
    gt_p: Union[str, os.PathLike],
    dt_p: Union[str, os.PathLike],
    iou_thres: float,
    category_ids: Union[List[int], Literal["all"]],
    ann_tags: Union[List[str], Literal["all"]],
    mode: Literal["bbox", "segm"],
    export_gt_eval_res_p: Union[str, os.PathLike],
    export_dt_eval_res_p: Union[str, os.PathLike],
) -> None:
    img_infos, category_infos, gt_infos = load_coco_gt(
        gt_p, category_ids, ann_tags
    )
    dt_infos = load_coco_dt(dt_p)

    if isinstance(category_ids, str) and category_ids == "all":
        category_ids = [c["id"] for c in category_infos]
    
    img_hws = [(i["height"], i["width"]) for i in img_infos]
    img_hws = np.asarray(img_hws, dtype=np.int_).reshape(-1, 2)

    # 找到最大的图片尺寸
    max_h = np.max(img_hws[:, 0]).item()
    max_w = np.max(img_hws[:, 1]).item()

    gt_category_ids = np.asarray([g["category_id"] for g in gt_infos])
    gt_img_ids = np.asarray([g["image_id"] for g in gt_infos])
    gt_img_hws = np.asarray([img_hws[g] for g in gt_img_ids]).reshape(-1, 2)

    dt_category_ids = np.asarray([d["category_id"] for d in dt_infos])
    dt_img_ids = np.asarray([d["image_id"] for d in dt_infos])
    dt_scores = np.asarray([d["score"] for d in dt_infos])
    dt_img_hws = np.asarray([img_hws[d] for d in dt_img_ids]).reshape(-1, 2)

    if mode == "bbox":
        gt_insts = np.asarray(g["bbox"] for g in gt_infos)
        dt_insts = np.asarray(d["bbox"] for d in dt_infos)
    elif mode == "segm":
        # 将polygon和rle转化为mask
        # 将mask统一pad为最大尺寸
        gt_insts = [g["segmentation"] for g in gt_infos]
        for i in range(len(gt_insts)):
            img_h, img_w = gt_img_hws[i]
            gt_poly = gt_insts[i]
            gt_mask = polyscoco2masks(gt_poly, (img_h, img_w), True)
            gt_insts[i] = center_pad_masks(gt_mask, (max_h, max_w))

        dt_insts = [d["segmentation"] for d in dt_insts]
        for i in range(len(dt_insts)):
            img_h, img_w = dt_img_hws[i]
            dt_rle = dt_insts[i]
            dt_mask = rles2masks(dt_rle)
            dt_insts[i] = center_pad_masks(dt_mask, (max_h, max_w))

        gt_insts = np.asarray(gt_insts)
        dt_insts = np.asarray(dt_insts)
    
    # 计算每个category的mAP
    for category_id in category_ids:
        gt_indice = gt_category_ids == category_id
        gt_category_ids_cati = gt_category_ids[gt_indice]
        gt_img_ids_cati = gt_img_ids[gt_indice]
        gt_img_hws_cati = gt_img_hws[gt_indice]
        gt_insts_cati = gt_insts[gt_indice]

        dt_indice = dt_category_ids == category_id
        dt_category_ids_cati = dt_category_ids[dt_indice]
        dt_img_ids_cati = dt_img_ids[dt_indice]
        dt_scores_cati = dt_scores[dt_indice]
        dt_insts_cati = dt_insts[dt_indice]

        