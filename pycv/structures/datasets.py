import os
from typing import List, Dict, Union

import numpy as np

from pycv.structures.insts import Insts


class DetDataset:
    """
    Attrs
    -----
    - `img_insts_ids`: `Dict[int, List[int]]`, `(num_imgs, (num_insts_per_img, ))`,
    `{img_id: [inst_id, ...]}`
    - `img_tags`: `Dict[int, List[str]]`, `(num_imgs, (num_img_tags, ))`,
    `{img_id: [img_tag, ...]}`
    - `img_ps`: `Dict[int, str]`, `(num_imgs, )`, `{img_id: img_p}`
    - `inst_img_ids`: `Dict[int, int]`, `(num_insts, )`, `{inst_id: img_id}`
    - `inst_tags`: `Dict[int, List[str]]`, `(num_insts, (num_inst_tags, ))`,
    `{inst_id: [inst_tag, ...]}`
    - `insts`: `Dict[int, Insts]`, `(num_insts, (1, ))`, `{inst_id: inst}`
    - `cat_id_name_dict`: `Dict[int, str]`, `(num_cats, )`, `{cat_id: cat_name}`
    - `cat_name_id_dict`: `Dict[str, int]`, `(num_cats, )`, `{cat_name: cat_id}`

    Methods
    -----
    - `concat`
    - `convert_bboxes_format`
    - `convert_masks_format`
    - `get_data_by_img_ids`
    - `get_data_by_inst_ids`
    - `get_data_by_cats`
    - `get_data_by_img_tags`
    - `get_data_by_inst_tags`
    """

    def __init__(
        self
    ) -> None:
        pass