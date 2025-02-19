import copy
import json
import os
from pathlib import Path
from typing import Union, Literal, Dict, List

import numpy as np
import pycocotools.mask as pycocomask

from pycv.structures import DetInsts, SegmInsts


def labelme2insts(
    labelme_p: Union[os.PathLike, str],
    cat_name_id_dict: Dict[str, int],
) -> Union[SegmInsts, DetInsts]:
    with open(labelme_p, "r") as f:
        labelme_dict = json.load(f)
    
    anns = labelme_dict["shapes"]
    cats = []
    masks = []
    bboxes = []

    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]

    for ann in anns:
        cat = cat_name_id_dict[ann["label"]]

        if ann["shape_type"] == "rectangle":
            x1y1, x2y2 = ann["points"]
            x1, y1 = x1y1
            x2, y2 = x2y2
            bbox = [x1, y1, x2, y2]
            mask = None
        elif ann["shape_type"] == "polygon":
            poly = np.asarray(ann["points"], dtype=np.int32) # (num_points, 2)
            poly = poly.flatten().tolist() # (num_points * 2, )
            rle = pycocomask.frPyObjects([poly], img_h, img_w)
            mask = pycocomask.decode(rle) # (img_h, img_w, 1)
            mask = np.transpose(mask, (2, 0, 1)) # (1, img_h, img_w)
            x1, y1, w, h = pycocomask.toBbox(rle).flatten().tolist()
            x2, y2 = x1 + w, y1 + h
            bbox = [x1, y1, x2, y2]

        bboxes.append(bbox)
        masks.append(mask)
        cats.append(cat)
    
    cats = np.asarray(cats, dtype=np.int32)
    scores = np.ones_like(cats, dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.int32)

    if masks[0] is not None:
        masks = np.concatenate(masks, axis=0)
        insts = SegmInsts(scores, cats, bboxes, masks)
    else:
        insts = DetInsts(scores, cats, bboxes)
    
    return insts


def labelmefolder2coco(
    labelme_folder: Union[str, os.PathLike],
    export_dir: Union[str, os.PathLike],
    export_json_name: str = "annotation.json",
    img_prefix: Union[str, Literal["", "abs"]] = "",
    cat_name_id_dict: Dict[str, int] = {},
) -> None:
    """
    Args:
        labelme_folder (Union[str, os.PathLike]`): 
            labelme folder path
        export_dir (Union[str, os.PathLike]): 
            export directory
        export_json_name (str): 
            export json name
        img_prefix (str): 
            one of the following:
            - ``""``: use img name as ``file_name``
            - ``"abs"``: use absolute img path as ``file_name``
            - other: use ``f"{img_prefix}/{img_name}"`` as ``file_name``
        cat_name_id_dict (dict): 
            Used to specify exported coco cat ids. 
            Categories not in the keys of this param will be ignored.
    """
    assert len(cat_name_id_dict) > 0

    coco_img_template = {
        "height": 0,
        "width": 0,
        "id": 0,
        "file_name": ""
    }

    coco_category_template = {
        "id": 0,
        "name": ""
    }

    coco_ann_template = {
        "id": 0,
        "iscrowd": 0,
        "image_id": 0,
        "area": 0,
        "bbox": [],
        "segmentation": [[]],
    }

    os.makedirs(export_dir, exist_ok=True)

    filenames = os.listdir(labelme_folder)
    filenames.sort()

    # Initialize img, category, annotation ids
    img_id = 0
    ann_id = 0

    # Initialize img, category, annotation list
    coco_img_list = []
    coco_category_list = []
    coco_ann_list = []

    include_cat_names = []
    cat_name_id_dict_ = {}
    cat_id_name_dict_ = {}
    
    for cat_name, cat_id in cat_name_id_dict.items():
        include_cat_names.append(cat_name)
        cat_name_id_dict_[cat_name] = cat_id
        cat_id_name_dict_[cat_id] = cat_name
        
    cat_name_id_dict = cat_name_id_dict_
    del cat_id_name_dict_, cat_name_id_dict_

    for filename in filenames:
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue

        if filename.startswith("seg_"):
            continue

        img_p = os.path.join(labelme_folder, filename)
        img_name = filename
        img_suffix = Path(filename).suffix
        labelme_name = filename.replace(img_suffix, ".json")
        labelme_p = os.path.join(labelme_folder, labelme_name)

        if not os.path.exists(labelme_p):
            continue

        with open(labelme_p, "r") as f:
            labelme_dict = json.load(f)

        # Add img to img_list
        img_h = labelme_dict["imageHeight"]
        img_w = labelme_dict["imageWidth"]

        if img_prefix == "":
            file_name = img_name
        elif img_prefix == "abs":
            file_name = img_p
        else:
            file_name = os.path.join(img_prefix, img_name)
        
        coco_img = copy.deepcopy(coco_img_template)
        coco_img["height"] = img_h
        coco_img["width"] = img_w
        coco_img["id"] = img_id
        coco_img["file_name"] = file_name

        coco_img_list.append(coco_img)
        img_id += 1

        # Ann buffer to process shapes with same group_id
        ann_buffer: Dict[Union[None, int], List[dict]] = {}
        group_id_single = 0

        # Put anns with the same group_id into one list
        for shape in labelme_dict["shapes"]:
            # skip this shape if its cat not in include_cat_names
            shape_cat_name = shape["label"]
            if shape_cat_name not in include_cat_names:
                continue

            group_id = shape["group_id"]

            if group_id is None:
                group_id = f"null{group_id_single}"
                group_id_single += 1

            if group_id not in ann_buffer.keys():
                ann_buffer[group_id] = []

            ann_buffer[group_id].append(shape)

        # Merge anns with the same group_id
        for group_id, group_anns in ann_buffer.items():
            if len(group_anns) == 1:
                # Non-occluded instance, 1 poly
                ann = group_anns[0]
                ann_category_name = ann["label"]
                polys = ann["points"]
                
                if ann["shape_type"] == "rectangle":
                    x1 = polys[0][0]
                    y1 = polys[0][1]
                    x2 = polys[1][0]
                    y2 = polys[1][1]
                    w = x2 - x1
                    h = y2 - y1
                    bbox = [x1, y1, w, h]
                    area = w * h
                    polys = None
                else:
                    polys = [np.asarray(polys).flatten().tolist()]
                    rle = pycocomask.frPyObjects(polys, img_h, img_w)
                    bbox = pycocomask.toBbox(rle).tolist()[0]
                    area = pycocomask.area(rle).item()
            else:
                # Occluded instance, 
                # 1 bbox + multiple polys, or multiple polys
                ann_category_name = group_anns[0]["label"]
                polys = []
                bbox = []

                for ann in group_anns:
                    if ann["shape_type"] == "polygon":
                        poly = ann["points"]
                        poly = np.asarray(poly).flatten().tolist()
                        polys.append(poly)
                    elif ann["shape_type"] == "rectangle":
                        bbox = ann["points"]
                        x1 = bbox[0][0]
                        y1 = bbox[0][1]
                        x2 = bbox[1][0]
                        y2 = bbox[1][1]
                        w = x2 - x1
                        h = y2 - y1
                        bbox = [x1, y1, w, h]
                
                rles = pycocomask.frPyObjects(polys, img_h, img_w)
                rle = pycocomask.merge(rles)
                area = pycocomask.area(rle).item()

                # if no rect labelled, compute bbox from poly
                if len(bbox) == 0:
                    polys = polys
                    bbox = pycocomask.toBbox(rle).tolist()
            
            ann_category_id = cat_name_id_dict[ann_category_name]

            coco_ann = copy.deepcopy(coco_ann_template)
            coco_ann["iscrowd"] = 0
            coco_ann["category_id"] = ann_category_id
            coco_ann["image_id"] = coco_img["id"]
            coco_ann["bbox"] = bbox
            coco_ann["area"] = area
            coco_ann["id"] = ann_id

            if polys is not None:
                coco_ann["segmentation"] = polys

            coco_ann_list.append(coco_ann)
            ann_id += 1
    
    # Add categories
    for cat_name, cat_id in cat_name_id_dict.items():
        cat_info = copy.deepcopy(coco_category_template)
        cat_info["id"] = cat_id
        cat_info["name"] = cat_name
        coco_category_list.append(cat_info)

    coco_ann = {
        "images": coco_img_list,
        "categories": coco_category_list,
        "annotations": coco_ann_list
    }

    # Export coco
    export_p = os.path.join(export_dir, export_json_name)
    with open(export_p, "w") as f:
        json.dump(coco_ann, f)