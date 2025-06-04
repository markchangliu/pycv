import os
from typing import List, Dict, Union, Literal

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
    - `convert_bbox_format`
    - `convert_mask_format`
    - `get_data_by_img_ids`
    - `get_data_by_inst_ids`
    - `get_data_by_cats`
    - `get_data_by_img_tags`
    - `get_data_by_inst_tags`
    """

    def __init__(
        self,
        img_insts_ids: Dict[int, List[int]],
        img_tags: Dict[int, List[str]],
        img_ps: Dict[int, str],
        inst_img_ids: Dict[int, int],
        inst_tags: Dict[int, List[str]],
        insts: Dict[str, Insts],
        cat_name_id_dict: Dict[str, int]
    ) -> None:
        assert len(img_insts_ids) == len(img_tags) == len(img_ps)
        assert len(inst_img_ids) == len(inst_tags) == len(inst_tags)
        
        self.img_insts_ids = img_insts_ids
        self.img_tags = img_tags
        self.img_ps = img_ps
        self.inst_img_ids = inst_img_ids
        self.inst_tags = inst_tags
        self.insts = insts
        self.cat_name_id_dict = cat_name_id_dict
        self.cat_id_name_dict = {v: k for k, v in cat_name_id_dict.items()}
    
    def convert_bbox_format(
        self,
        dst_format: Literal["XYXY", "XYWH"]
    ) -> None:
        for inst in self.insts.values():
            inst.bboxes.convert_format(dst_format)
    
    def convert_mask_format(
        self,
        dst_format: Literal["polygon", "binary", "rle"]
    ) -> None:
        for inst in self.insts.values():
            if inst.masks is not None:
                inst.masks.convert_format(dst_format)
    
def concat_datasets()