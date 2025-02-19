import os
import shutil
from pathlib import Path
from typing import Union, Literal

import numpy as np
from pycocotools.coco import COCO

from pycv.structures import DetInsts, SegmInsts
from pycv.labels.convert.insts import insts2labelme


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