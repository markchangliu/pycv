import json
import os
from pathlib import Path
from typing import List, Dict, Union

import cv2
import numpy as np

from pycv.data_structures.bboxes import BBoxes, BBoxFormat
from pycv.data_structures.masks import Masks, MaskFormat
from pycv.data_structures.insts import Insts
from pycv.data_structures.det_data import DetData
from pycv.datasets.det_datasets import DetDataset
from pycv.datasets.label_parsers import parser_labelme
from pycv.io import load_files


def load_dataset_from_labelme_dirs(
    img_dirs: List[str, os.PathLike],
    labelme_dirs: List[str, os.PathLike],
    cat_name_id_dict: Dict[str, int]
) -> DetDataset:
    """
    Args
    - `img_dirs`: `List[str, os.PathLike]`, `(num_ds, )`
    - `labelme_dirs`: `List[str, os.PathLike]`, `(num_ds, )`
    - `cat_name_id_dict`: `Dict[str, int]`

    Return
    - `det_dataset`: `DetDataset`
    """
    cat_id_name_dict = {v:k for k, v in cat_name_id_dict.items()}

    data_list = []
    img_ids = []
    insts_ids = []

    curr_img_id = 0
    curr_inst_id = 0

    for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
        img_generator = load_files(img_dir, include_suffixes=(".png", ".jpg", ".jpeg"))

        for img_p in img_generator:
            img_dir = Path(img_p).parent
            img_stem = Path(img_p).stem
            labelme_p = os.path.join(labelme_dir, f"{img_stem}.json")

            if not os.path.exists(labelme_p):
                continue
            
            det_data = parser_labelme(
                labelme_p, cat_name_id_dict, img_dir
            )

            data_list.append(det_data)
            img_ids.append(curr_img_id)
            insts_ids.append(
                list(range(curr_inst_id, curr_inst_id + len(det_data)))
            )

            curr_img_id += 1
            curr_inst_id += len(det_data)

    det_dataset = DetDataset(
        data_list, cat_id_name_dict, cat_name_id_dict, img_ids, insts_ids
    )

    return det_dataset

def load_dataset_from_coco_json(
    coco_p: Union[str, os.PathLike],
    img_prefix: Union[str, os.PathLike]
) -> DetDataset:
    with open(coco_p, "r") as f:
        coco_dict = json.load(f)
    
    cat_name_id_dict = {}
    cat_id_name_dict = {}

    for cat_info in coco_dict["categories"]:
        cat_name = cat_info["name"]
        cat_id = cat_info["id"]
        cat_name_id_dict[cat_name] = cat_id
        cat_id_name_dict[cat_id] = cat_name
    
    img_ps = []
    img_ids = []
    img_hws = []
    imgs_tags = []
    
    for img_info in coco_dict["images"]:
        img_p = os.path.join(img_prefix, img_info["file_name"])
        img_id = img_info["id"]
        img_hw = (img_info["height"], img_info["width"])
        img_tags = img_info.get("tags", [])
        img_ps.append(img_p)
        img_ids.append(img_id)
        img_hws.append(img_hw)
        imgs_tags.append(img_tags)
    
    imgs_bboxes = [[] * max(img_ids)]
    imgs_masks = [[] * max(img_ids)]
    imgs_cat_ids = [[] * max(img_ids)]
    imgs_confs = [[] * max(img_ids)]
    imgs_insts_tags = [[] * max(img_ids)]
    imgs_insts_ids = [[] * max(img_ids)]

    for ann_info in coco_dict["annotations"]:
        conf = 1
        cat_id = ann_info["category_id"]
        bbox = ann_info["bbox"]
        segm = ann_info.get(["segmentation"], None)
        ann_img_id = ann_info["image_id"]
        inst_tags = ann_info.get("tags", [])
        ann_id = ann_info["id"]

        imgs_masks[ann_img_id].append(segm)
        imgs_bboxes[ann_img_id].append(bbox)
        imgs_cat_ids[ann_img_id].append(cat_id)
        imgs_confs[ann_img_id].append(conf)
        imgs_insts_tags[ann_img_id].append(inst_tags)
        imgs_insts_ids[ann_img_id].append(ann_id)
    
    data_list = []

    for i in range(len(img_ids)):
        img_id = img_ids[i]
        img_p = img_ps[i]
        img_hw = img_hws[i]
        img_tags = imgs_tags[i]

        img_bboxes = imgs_bboxes[img_id]
        img_masks = imgs_masks[img_id] if imgs_masks[img_id][0] else None
        img_cat_ids = imgs_cat_ids[img_id]
        img_confs = imgs_confs[img_id]
        img_insts_tags = imgs_insts_tags[img_id]

        img_bboxes = BBoxes(img_bboxes, BBoxFormat.XYWH)
        img_masks = Masks(img_masks, img_hw, MaskFormat.POLY) if img_masks else None
        img_confs = np.asarray(img_confs)
        img_cat_ids = np.asarray(img_cat_ids)
        img_tags = set(img_tags)
        insts_tags = [set(ts) for ts in img_insts_tags]
        insts = Insts(img_confs, img_cat_ids, img_bboxes, img_masks)
        img_data = DetData(img_p, insts, img_tags, insts_tags)

        data_list.append(img_data)
    
    img_dataset = DetDataset(
        data_list, cat_id_name_dict, cat_name_id_dict,
        img_ids, imgs_insts_ids
    )

    return img_dataset

def load_dataset_from_coco_jsons(
    coco_ps: List[Union[os.PathLike, str]],
    img_prefixes: List[Union[os.PathLike, str]],
) -> DetDataset:
    datasets: List[DetDataset] = []
    
    for coco_p, img_prefix in zip(coco_ps, img_prefixes):
        ds = load_dataset_from_coco_json(coco_p, img_prefix)
        datasets.append(ds)
    
    dataset = datasets[0].concat(datasets[1:])

    return dataset


        
    
