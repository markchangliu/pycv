import os
from typing import List, Dict, Union, Literal, Tuple

import numpy as np

from pycv.structures.insts import Insts


class DetDataset:
    """
    Attrs (data)
    -----
    - `img_tags`: `Dict[int, List[str]]`, `(num_imgs, (num_img_tags, ))`,
    `{img_id: [img_tag, ...]}`
    - `img_ps`: `Dict[int, str]`, `(num_imgs, )`, `{img_id: img_p}`
    - `inst_tags`: `Dict[int, List[str]]`, `(num_insts, (num_inst_tags, ))`,
    `{inst_id: [inst_tag, ...]}`
    - `insts`: `Dict[int, Insts]`, `(num_insts, (1, ))`, `{inst_id: inst}`
    - `cat_id_name_dict`: `Dict[int, str]`, `(num_cats, )`, `{cat_id: cat_name}`
    - `cat_name_id_dict`: `Dict[str, int]`, `(num_cats, )`, `{cat_name: cat_id}`

    Attrs (index)
    -----
    - `img_insts_ids`: `Dict[int, List[int]]`, `(num_imgs, (num_insts_per_img, ))`,
    `{img_id: [inst_id, ...]}`
    - `inst_img_ids`: `Dict[int, int]`, `(num_insts, )`, `{inst_id: img_id}`
    - `img_tag_img_ids`: `Dict[str, List[int]]`, `(num_img_tags, (num_imgs_per_tag))`,
    `{img_tag: [img_id, ...]}`
    - `inst_tag_inst_ids`: `Dict[str, List[int]]`, `(num_inst_tags, (num_insts_per_tag))`,
    `{inst_tag: [inst_id, ...]}`
    - `cat_id_img_ids`: `Dict[int, List[int]]`, `(num_cats, (num_imgs_per_cat))`,
    `{cat_id: [img_id, ...]}`
    - `cat_id_inst_ids`: `Dict[int, List[int]]`, `(num_cats, (num_insts_per_cat))`,
    `{cat_id: [inst_id, ...]}`

    Methods
    -----
    - `concat`
    - `convert_bbox_format`
    - `convert_mask_format`
    - `get_subset_by_img_ids`
    - `get_subset_by_inst_ids`
    - `get_subset_by_cat_ids`
    - `get_subset_by_cat_names`
    - `get_subset_by_img_tags`
    - `get_subset_by_inst_tags`
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
        
        self.img_tags = img_tags
        self.img_ps = img_ps
        self.inst_tags = inst_tags
        self.insts = insts
        self.cat_name_id_dict = cat_name_id_dict
        self.cat_id_name_dict = {v: k for k, v in cat_name_id_dict.items()}

        # build index
        self.img_insts_ids = img_insts_ids
        self.inst_img_ids = inst_img_ids
        self.img_tag_img_ids: Dict[str, List[int]] = {}
        self.inst_tag_inst_ids: Dict[str, List[int]] = {}
        self.cat_id_img_ids: Dict[int, List[int]] = {}
        self.cat_id_inst_ids: Dict[int, List[int]] = {}

        for img_id, img_tags in self.img_tags.items():
            for t in img_tags:
                if t in self.img_tag_img_ids.keys():
                    self.img_tag_img_ids[t].append(img_id)
                else:
                    self.img_tag_img_ids[t] = [img_id]
        
        for inst_id, inst_tags in self.inst_tags.items():
            for t in inst_tags:
                if t in self.inst_tag_inst_ids.keys():
                    self.inst_tag_inst_ids[t].append(inst_id)
                else:
                    self.inst_tag_inst_ids[t] = [inst_id]
        
        for inst_id, inst in self.insts.items():
            cat_id = inst.cat_ids.item()
            img_id = self.inst_img_ids[inst_id]

            if cat_id in self.cat_id_img_ids.keys():
                self.cat_id_img_ids[cat_id].append(img_id)
            else:
                self.cat_id_img_ids[cat_id] = [img_id]
    
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
    ) -> None:
        all_datasets = [self] + datasets
        new_dataset = concat_datasets(*all_datasets)
        
        self.cat_id_name_dict = new_dataset.cat_id_name_dict
        self.cat_name_id_dict = new_dataset.cat_name_id_dict
        self.img_insts_ids = new_dataset.img_insts_ids
        self.img_ps = new_dataset.img_ps
        self.img_tags = new_dataset.img_tags
        self.inst_img_ids = new_dataset.inst_img_ids
        self.inst_tags = new_dataset.inst_tags
        self.insts = new_dataset.insts
    
    def get_subset_by_img_ids(
        self,
        *img_ids: int,
    ) -> "DetDataset":
        new_img_ids = img_ids
        new_inst_img_ids = {k: v for k, v in self.inst_img_ids if v in new_img_ids}
        new_img_insts_ids = {k: v for k, v in self.img_insts_ids if k in new_img_ids}

        new_img_tag_img_ids = {}
        for k, v in self.img_tag_img_ids



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
        