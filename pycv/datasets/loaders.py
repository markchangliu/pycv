import json
import os
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

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
    inst_ids = []

    curr_img_id = 0
    curr_inst_id = 0

    for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
        img_generator = load_files(img_dir, include_suffixes=(".png", ".jpg", ))
        
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
            inst_ids.append(
                list(range(curr_inst_id, curr_inst_id + len(det_data)))
            )

            curr_img_id += 1
            curr_inst_id += len(det_data)

    det_dataset = DetDataset(
        data_list, cat_id_name_dict, cat_name_id_dict, img_ids
    )

    return det_dataset
