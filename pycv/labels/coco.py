import os
import json
import shutil
from pathlib import Path
from typing import Union, Literal, List, Tuple, Dict

import numpy as np
from pycocotools.coco import COCO

from pycv.structures import DetInsts, SegmInsts
from pycv.labels.insts import insts2labelme


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
) -> Tuple[dict, dict, dict]:
    """
    Args
    - `coco_gt_p`: `Union[str, os.PathLike]`
    - `category_ids`: `Union[Literal["all"], List[int]]`, 要包含的category id, 
    `"all"`表示所有
    - `ann_tags`: `Union[Literal["all"], List[str]]`, 要包含的ann tag, 
    `"all"`表示所有

    Returns
    - `img_id_info_dict`: `Dict[int, dict]`
        - info keys `height`, `width`, `id`, `file_name`
    - `category_id_info_dict`: `Dict[int, dict]`
        - info keys `id`, `name`
    - `ann_id_info_dict`: `Dict[int, dict]`
        - info keys `id`, `iscrowd`, `image_id`, `area`, `bbox`, 
        `segmentation`, `category_id`, `ann_tags`
    """
    with open(coco_gt_p, "r") as f:
        coco_gt = json.load(f)

    imgs: List[dict] = coco_gt["images"]
    categories: List[dict] = coco_gt["categories"]
    anns: List[dict] = coco_gt["annotations"]

    if isinstance(category_ids, list):
        categories = [c for c in categories if c["id"] in category_ids]
    elif isinstance(category_ids, str) and category_ids == "all":
        category_ids = [c["id"] for c in categories]
    
    # 过滤掉不要的category_id
    img_id_info_dict = {i["id"]: i for i in imgs}
    category_id_info_dict = {c["id"]: c for c in categories if c["id"] in category_ids}
    ann_id_info_dict = {a["id"]: a for a in anns if a["category_id"] in category_ids}

    if isinstance(ann_tags, str) and ann_tags == "all":
        return img_id_info_dict, category_id_info_dict, category_id_info_dict
    
    # 过滤掉不要的tag
    for ann_id, ann_dict in ann_id_info_dict.items():
        ann_tags = ann_dict["tags"]
        exclude_flag = True

        for t in ann_tags:
            if t in ann_tags:
                exclude_flag = False
                break
        
        if exclude_flag:
            del ann_id_info_dict[ann_id]

    return img_id_info_dict, category_id_info_dict, category_id_info_dict


def load_coco_dt(
    coco_dt_p: Union[str, os.PathLike],
    category_ids: Union[Literal["all"], List[str]],
) -> dict:
    """
    Args
    - `coco_dt_p`: `Union[str, os.PathLike]`
    - `category_ids`: `Union[Literal["all"], List[int]]`, 要包含的category id, 
    `"all"`表示所有

    Returns
    - `dt_id_info_dict`: `Dict[int, dict]`
        - info keys `image_id`, `category_id`, `bbox`, 
        `segmentation`, `score`
    """
    with open(coco_dt_p, "r") as f:
        coco_dt: List[dict] = json.load(f)
    
    if isinstance(category_ids, str) and category_ids == "all":
        dt_id_info_dict = {k: v for k, v in enumerate(coco_dt)}
        return dt_id_info_dict
    
    # 过滤掉不要的category_id
    dt_id_info_dict = {
        k: v for k, v in enumerate(coco_dt) if v["category_id"] in category_ids
    }

    return dt_id_info_dict


def get_anns_of_img(
    ann_id_info_dict: Dict[int, dict],
    img_id: int,
) -> List[dict]:
    """
    Args
    - `ann_id_info_dict`: `Dict[int, dict]`, info keys `id`, 
    `iscrowd`, `image_id`, `area`, `bbox`, `segmentation`, `category_id`, `ann_tags`
    - `img_id`: int

    Returns
    - `anns_of_img`: `List[dict]`
    """
    anns_of_img = [
        v for v in ann_id_info_dict.values() if v["image_id"] == img_id
    ]

    return anns_of_img


def get_dts_of_img(
    dt_id_info_dict: Dict[int, dict],
    img_id: int,
) -> List[dict]:
    """
    Args
    - `dt_id_info_dict`: `Dict[int, dict]`, info keys `image_id`, 
    `category_id`, `bbox`, `segmentation`, `score`
    - `img_id`: int

    Returns
    - `dts_of_img`: `List[dict]`
    """
    dts_of_img = [
        v for v in dt_id_info_dict.values() if v["image_id"] == img_id
    ]

    return dts_of_img