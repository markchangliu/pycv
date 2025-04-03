import os
from dataclasses import dataclass
from typing import Union, List, Set

import numpy as np

from pycv.data_structures.base import BaseStructure
from pycv.data_structures.insts import Insts


@dataclass
class DetData(BaseStructure):
    img: Union[np.ndarray, str, os.PathLike]
    insts: Insts
    img_tags: List[Set[str]] # (num_img_tags, )
    insts_tags: List[Set[str]] # (num_insts, (num_inst_tags, ))

    def __post_init__(self) -> None:
        self.validate()
    
    def __len__(self) -> int:
        return len(self.insts)

    def validate(self):
        if isinstance(self.img, str) and self.insts.masks is not None:
            img_h, img_w = self.img.shape[:2]
            mask_img_h, mask_img_w = self.insts.masks.img_hw
            
            if img_h != mask_img_h or img_w != mask_img_w:
                raise ValueError("img and insts.mask have different sizes")
    
    def get_insts_of_tags(
        self,
        target_insts_tags: List[str],
        return_indice_flag: bool
    ) -> "DetData":
        new_insts_idx = []
        target_insts_tags = set(target_insts_tags)
        
        for inst_id, inst_tags in enumerate(self.insts_tags):
            
            if inst_tags.intersection(target_insts_tags):
                new_insts_idx.append(inst_id)
        
        new_insts = self.insts[new_insts_idx]
        new_insts_tags = [self.insts_tags[i] for i in new_insts_idx]

        new_data  = DetData(
            self.img, new_insts, self.img_tags, new_insts_tags
        )

        if return_indice_flag:
            res = (new_data, new_insts_idx)
        else:
            res = new_data

        return new_data
            
    def get_insts_of_cat_ids(
        self,
        target_cat_ids: List[int]
    ) -> "DetData":
        target_cat_ids = np.asarray(target_cat_ids)
        inst_cat_ids = self.insts.cat_ids
        
        commons = np.intersect1d(
            inst_cat_ids, target_cat_ids, return_indices=True
        )

        new_insts_ids = commons[1]
        new_insts = self.insts[new_insts_ids]
        new_insts_tags = [self.insts_tags[i] for i in new_insts_ids]
        new_data = DetData(self.img, new_insts, self.img_tags, new_insts_tags)

        return new_data
