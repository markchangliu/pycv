import copy
import os
from dataclasses import dataclass
from typing import Union, List, Set, Tuple, Callable, Dict

import numpy as np

from pycv.data_structures.insts import Insts


@dataclass
class DetData:
    img: Union[str, os.PathLike]
    insts: Insts
    img_tags: Set[str] # (num_img_tags, )
    insts_tags: List[Set[str]] # (num_insts, (num_inst_tags, ))
    
    def __len__(self) -> int:
        return len(self.insts)
    
    def __getitem__(
        self, 
        item: Union[int, List[int], slice, np.ndarray]
    ) -> "DetData":
        if isinstance(item, int):
            item = [item]
        if isinstance(item, slice):
            item = list(range(len(self.insts))[item])
        
        new_insts = self.insts[item]
        new_insts_tags = [self.insts_tags[i] for i in item]
        new_data = DetData(self.img, new_insts, new_insts_tags)

        return new_data
    
    def get_subdata_of_inst_tags(
        self,
        target_inst_tags: List[str],
        return_idx_flag: bool
    ) -> Union["DetData", Tuple["DetData", List[int]]]:
        new_inst_ids = []
        target_inst_tags = set(target_inst_tags)
        
        for inst_id, inst_tags in enumerate(self.insts_tags):
            
            if inst_tags.intersection(target_inst_tags):
                new_inst_ids.append(inst_id)
        
        new_insts = self.insts[new_inst_ids]
        new_inst_tags = [self.insts_tags[i] for i in new_inst_ids]

        new_data  = DetData(
            self.img, new_insts, self.img_tags, new_inst_tags
        )

        if return_idx_flag:
            res = (new_data, new_inst_ids)
        else:
            res = new_data

        return new_data
            
    def get_subdata_of_cat_ids(
        self,
        target_cat_ids: List[int],
        return_idx_flag: bool
    ) -> Union["DetData", Tuple["DetData", List[int]]]:
        target_cat_ids = np.asarray(target_cat_ids)
        inst_cat_ids = self.insts.cat_ids
        
        commons = np.intersect1d(
            inst_cat_ids, target_cat_ids, return_indices=True
        )

        new_inst_ids = commons[1]
        new_insts = self.insts[new_inst_ids]
        new_inst_tags = [self.insts_tags[i] for i in new_inst_ids]
        new_data = DetData(self.img, new_insts, self.img_tags, new_inst_tags)

        if return_idx_flag:
            res = (new_data, new_inst_ids)
        else:
            res = new_data

        return res
    
    def tag_img(
        self,
        new_tags: List[str],
        lambda_funcs: List[Callable[[Union[str, os.PathLike]], bool]],
        retag_flag: bool
    ) -> None:
        img_tags = self.img_tags
        new_img_tags = [] if retag_flag else copy.deepcopy(img_tags)

        for new_tag, lambda_func in zip(new_tags, lambda_funcs):
            if lambda_func(self.img):
                new_img_tags.append(new_tag)
        
        self.img_tags = set(new_img_tags)
    
    def tag_insts(
        self,
        new_tags: List[str],
        lambda_funcs: List[Callable[[Insts], Union[np.ndarray, List[bool]]]],
        retag_flag: bool
    ) -> None:
        insts_tags = [list(ts) for ts in self.insts_tags]
        new_insts_tags = [[] * len(self.insts)] if retag_flag else copy.deepcopy(insts_tags)

        for new_tag, lambda_func in zip(new_tags, lambda_funcs):
            retag_flags = lambda_func(self.insts)

            for inst_id, retag_flag in enumerate(retag_flags):
                if retag_flag:
                    new_insts_tags[inst_id].append(new_tag)

        self.insts_tags = [set(ts) for ts in new_insts_tags]
    
    def update_cat_ids(
        self, 
        cat_id_old_new_dict: Dict[int, int]
    ) -> None:
        old_cat_ids = self.insts.cat_ids
        new_cat_ids = [cat_id_old_new_dict[c] for c in old_cat_ids]
        self.insts.cat_ids = np.asarray(new_cat_ids)
