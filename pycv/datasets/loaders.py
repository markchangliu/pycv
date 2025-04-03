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
    
    imgs = []
    img_ids = []
    
    for img_info in coco_dict["images"]:
        img_p = os.path.join(img_prefix, img_info["file_name"])
        img_id = img_info["id"]
        imgs.append(img_p)
        img_ids.append(img_id)
    
    anns = []
    ann_ids = [[] * (len(img_ids) + 1)]
    for ann_info in coco_dict["annotations"]:
        conf = 1
        cat_id = ann_info["category_id"]
        bbox = ann_info["bbox"]
        segm = ann_info["segmentation"]
        ann_img_id = ann_info["image_id"]

        
    
