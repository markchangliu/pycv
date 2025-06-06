import os
from typing import List, Dict, Union, Literal, Tuple

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
        insts: Dict[int, Insts],
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
    
    def concat(
        self,
        *datasets: "DetDataset"
    ) -> "DetDataset":
        all_datasets = [self] + datasets
        new_datasets = concat_datasets(*all_datasets)
        return new_datasets


def concat_datasets(*datasets: DetDataset) -> DetDataset:
    curr_img_id = 0
    curr_inst_id = 0
    # {ds_index, {old_img_id, new_img_id}}
    img_id_ds_old_new_dict: Dict[int, Dict[int, int]] = {}
    # {ds_index, {old_inst_id, new_inst_id}}
    inst_id_ds_old_new_dict: Dict[int, Dict[int, int]] = {}
    
    for i, d in enumerate(datasets):
        old_img_ids = list(d.img_ps.keys())
        new_img_ids = list(range(curr_img_id, curr_img_id + len(old_img_ids)))
        mapping = {old_img_ids[k]: new_img_ids[k] for k in range(len(old_img_ids))}
        img_id_ds_old_new_dict[i] = mapping
        curr_img_id += len(old_img_ids)

        old_inst_ids = list(d.insts.keys())
        new_inst_ids = list(range(curr_inst_id, curr_inst_id + len(old_inst_ids)))
        mapping = {old_inst_ids[k]: new_inst_ids[k] for k in range(len(old_inst_ids))}
        inst_id_ds_old_new_dict[i] = mapping
        curr_inst_id += len(old_inst_ids)
    
    new_cat_name_id_dict = {}
    curr_cat_id = 0
    for d in datasets:
        for k in d.cat_name_id_dict.keys():
            if k in new_cat_name_id_dict.keys():
                continue
            new_cat_name_id_dict[k] = curr_cat_id
            curr_cat_id += 1

    new_img_tags = {}
    new_img_ps = {}
    new_img_insts_ids = {}
    new_inst_img_ids = {}
    new_inst_tags = {}
    new_insts = {}

    for i, d in enumerate(datasets):
        img_id_old_new_dict = img_id_ds_old_new_dict[i]
        inst_id_old_new_dict = inst_id_ds_old_new_dict[i]

        new_img_tags.update({
            img_id_old_new_dict[k]: d.img_tags[k] \
            for k in img_id_old_new_dict.keys()
        })
        new_img_ps.update({
            img_id_old_new_dict[k]: d.img_ps[k] \
            for k in img_id_old_new_dict.keys()
        })
        
        img_insts_ids = {}
        for k, v in d.img_insts_ids.items():
            new_v = [inst_id_old_new_dict[s] for s in v]
            img_insts_ids[k] = new_v
        
        new_img_insts_ids.update({
            img_id_old_new_dict[k]: img_insts_ids[k] \
            for k in img_id_old_new_dict.keys()
        })

        inst_img_ids = {}
        for k, v in d.inst_img_ids.items():
            new_v = img_id_old_new_dict[v]
            inst_img_ids[k] = new_v
        
        new_inst_img_ids.update({
            inst_id_old_new_dict[k]: inst_img_ids[k] \
            for k in inst_id_old_new_dict.keys()
        })

        new_inst_tags.update({
            inst_id_old_new_dict[k]: d.inst_tags[k] \
            for k in inst_id_old_new_dict.keys()
        })

        new_insts.update({
            inst_id_old_new_dict[k]: d.insts[k] \
            for k in inst_id_old_new_dict.keys()
        })
    
    new_dataset = DetDataset(
        new_img_insts_ids,
        new_img_tags,
        new_img_ps,
        new_inst_img_ids,
        new_inst_tags,
        new_insts,
        new_cat_name_id_dict
    )

    return new_dataset
        