import copy
import json
import os
from pathlib import Path
from typing import Union, Literal, Dict, List, Any

import cv2


def yolo2coco(
    yolo_root: Union[str, os.PathLike],
    export_json_dir: Union[str, os.PathLike],
    coco_img_prefix: Literal["abs", "rel"],
    cat_id_name_dict: Dict[int, str],
    ann_mode: Literal["bbox", "mask"]
) -> None:
    """
    Args
    - `yolo_root`: `Union[str, os.PathLike]`, directory structure:
        - `images`:
            - `subdir1`
            - `subdir2`
            - ...
        - `labels`: same as `images`
    - `export_json_dir`: `Union[str, os.PathLike]`
    - `coco_img_prefix`: `Literal["abs", "rel"]`:
        - `abs`: absolute path
        - `rel`: relative to `{yolo_root}/images`
    - `cat_id_name_dict`: `Dict[int, str]`
    """
    img_dir = os.path.join(yolo_root, "images")
    yolo_ann_dir = os.path.join(yolo_root, "labels")

    subdir_names = os.listdir(img_dir)
    subdir_names.sort()

    coco_img_template = {
        "height": -1,
        "width": -1,
        "id": -1,
        "file_name": ""
    }

    coco_cat_template = {
        "id": -1,
        "name": ""
    }

    coco_ann_template = {
        "id": -1,
        "iscrowd": 0,
        "image_id": -1,
        "category_id": -1,
        "bbox": [],
        "segmentation": [[]],
        "area": 0
    }

    for subdir_name in subdir_names:
        img_subdir = os.path.join(img_dir, subdir_name)
        ann_subdir = os.path.join(yolo_ann_dir, subdir_name)

        if not os.path.isdir(img_dir):
            continue
        if not os.path.exists(ann_subdir):
            continue

        coco_imgs: List[Dict[str, Any]] = []
        coco_cats: List[Dict[str, Any]] = []
        coco_anns: List[Dict[str, Any]] = []

        img_id = 0
        ann_id = 0

        cat_name_id_dict: Dict[str, int] = {v: k for k, v in cat_id_name_dict.items()}

        file_names = os.listdir(img_subdir)
        file_names.sort()

        for file_name in file_names:
            if not file_name.endswith((".png", ".jpg", ".jpeg")):
                continue

            img_name = file_name
            img_p = Path(os.path.join(img_subdir, img_name))
            img_stem = img_p.stem
            yolo_ann_name = f"{img_stem}.txt"
            yolo_ann_p = os.path.join(ann_subdir, yolo_ann_name)

            if not os.path.exists(yolo_ann_p):
                continue
            
            img = cv2.imread(img_p)
            img_h, img_w = img.shape[:2]

            coco_img = copy.deepcopy(coco_img_template)
            coco_img["height"] = img_h
            coco_img["width"] = img_w
            coco_img["id"] = img_id

            if coco_img_prefix == "abs":
                coco_img["file_name"] = str(img_p)
            elif coco_img_prefix == "rel":
                coco_img["file_name"] = str(img_p.relative_to(yolo_root))
            
            yolo_anns: List[List[str]] = []
            with open(yolo_ann_p, "r") as f:
                for l in f:
                    l = l.strip().split()
                    
                    if len(l) == 0:
                        continue

                    yolo_anns.append(l)

            for yolo_ann in yolo_anns:
                if ann_mode == "bbox":
                    # normalized (cat_id, x_ctr, y_ctr, w, h)
                    try:
                        cat_id, x_ctr, y_ctr, w, h = yolo_ann
                        cat_id = int(cat_id)
                    except:
                        print("error yolo_ann:", yolo_ann)
                        continue

                    if cat_id not in cat_id_name_dict.keys():
                        continue

                    x_ctr = float(x_ctr) * img_w
                    y_ctr = float(y_ctr) * img_h
                    w = float(w) * img_w
                    h = float(h) * img_h

                    x1 = x_ctr - 0.5 * w
                    y1 = y_ctr - 0.5 * h
                    bbox = [x1, y1, w, h]
                    bbox = [int(i) for i in bbox]

                elif ann_mode == "mask":
                    # normalized (cat_id, x1, y1, ..., xn, yn)
                    raise NotImplementedError

                area = bbox[2] * bbox[3]

                coco_ann = copy.deepcopy(coco_ann_template)
                coco_ann["id"] = ann_id
                coco_ann["image_id"] = img_id
                coco_ann["category_id"] = cat_id
                coco_ann["bbox"] = bbox
                coco_ann["area"] = area

                coco_anns.append(coco_ann)
                ann_id += 1
            
            coco_imgs.append(coco_img)
            img_id += 1

            if img_id % 100 == 0:
                print(f"complete {img_id + 1} images")

        for k, v in cat_id_name_dict.items():
            coco_cat = copy.deepcopy(coco_cat_template)
            coco_cat["id"] = k
            coco_cat["name"] = v
            coco_cats.append(coco_cat)

        coco_json = {
            "images": coco_imgs,
            "categories": coco_cats,
            "annotations": coco_anns
        }

        export_json_p = os.path.join(export_json_dir, f"{subdir_name}.json")
        with open(export_json_p, "w") as f:
            json.dump(coco_json, f)

        print(f"export {export_json_p}")