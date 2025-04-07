import os
from dataclasses import dataclass
from typing import List, Dict, Union, Callable

import numpy as np

from pycv.data_structures.insts import Insts
from pycv.data_structures.det_data import DetData


@dataclass
class DetDataset:
    data_list: List[DetData] # (num_imgs, )
    cat_id_name_dict: Dict[int, str] # (num_cats, )
    cat_name_id_dict: Dict[str, int] # (num_cats, )
    img_ids: List[int] # (num_imgs, )
    insts_ids: List[List[int]] # (num_imgs, (num_insts_per_img, ))

    def __post_init__(self) -> None:
        self.validate()
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(
        self, 
        item: Union[int, List[int], slice, np.ndarray]
    ) -> "DetDataset":
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, slice):
            item = list(range(len(self.insts))[item])
        elif isinstance(item, np.ndarray):
            if item.dtype == np.bool_:
                item = np.nonzero(item)[0].tolist()
            else:
                item = item.tolist()
        
        new_data_list = [self.data_list[i] for i in item]
        new_img_ids = item
        new_insts_ids = [self.insts_ids[i] for i in item]
        new_dataset = DetDataset(
            new_data_list, self.cat_id_name_dict, self.cat_name_id_dict,
            new_img_ids, new_insts_ids
        )

        return new_dataset

    def get_subset_of_cat_ids(
        self, 
        target_cat_ids: List[int]
    ) -> "DetDataset":
        new_cat_id_name_dict = {
            k: v for k, v in self.cat_id_name_dict if k in target_cat_ids
        }
        new_cat_name_id_dict = {
            k: v for k, v in self.cat_name_id_dict if v in target_cat_ids
        }

        new_data_list = []
        new_img_ids = []
        new_insts_ids = []

        for img_id, inst_ids, data in zip(self.img_ids, self.insts_ids, self.data_list):
            new_data, new_inst_ids_local = data.get_insts_of_cat_ids(target_cat_ids, True)
            
            if len(new_inst_ids_local) == 0:
                continue
            
            new_inst_ids = [inst_ids[i] for i in new_inst_ids_local]

            new_data_list.append(new_data)
            new_img_ids.append(img_id)
            new_insts_ids.append(new_inst_ids)
        
        new_dataset = DetDataset(
            new_data_list, new_cat_id_name_dict, new_cat_name_id_dict,
            new_img_ids, new_insts_ids
        )

        return new_dataset
    
    def get_subset_of_img_tags(
        self,
        target_img_tags: List[str]
    ) -> "DetDataset":
        target_img_tags = set(target_img_tags)
        new_data_list = []
        new_img_ids = []
        new_insts_ids = []

        for img_id, inst_ids, data in zip(self.img_ids, self.insts_ids, self.data_list):
            if data.img_tags.intersection(target_img_tags):
                new_data_list.append(data)
                new_img_ids.append(img_id)
                new_insts_ids.append(inst_ids)
        
        new_dataset = DetDataset(
            new_data_list, self.cat_id_name_dict, self.cat_name_id_dict,
            new_img_ids, new_insts_ids
        )

        return new_dataset
    
    def get_subset_of_inst_tags(
        self,
        target_inst_tags: List[str],
    ) -> "DetDataset":
        target_inst_tags = set(target_inst_tags)
        new_data_list = []
        new_img_ids = []
        new_insts_ids = []

        for img_id, inst_ids, data in zip(self.img_ids, self.insts_ids, self.data_list):
            new_data, new_inst_ids_local = data.get_insts_of_tags(target_inst_tags, True)

            if len(new_inst_ids_local) == 0:
                continue

            new_inst_ids = [inst_ids[i] for i in new_inst_ids_local]

            new_data_list.append(new_data)
            new_img_ids.append(img_id)
            new_insts_ids.append(new_inst_ids)
            
        new_dataset = DetDataset(
            new_data_list, self.cat_id_name_dict, self.cat_name_id_dict,
            new_img_ids, new_insts_ids
        )

        return new_dataset

    def get_subset_of_img_ids(
        self,
        target_img_ids: List[int]
    ) -> "DetDataset":
        pass
        
    def reindex(self) -> None:
        
        pass
    
    def tag_imgs(
        self,
        new_tags: List[str],
        lambda_funcs: List[Callable[[Union[str, os.PathLike, np.ndarray]], bool]],
        retag_flag: bool
    ) -> None:
        for data in self.data_list:
            data.tag_img(new_tags, lambda_funcs, retag_flag)
    
    def tag_insts(
        self,
        new_tags: List[str],
        lambda_funcs: List[Callable[[Insts], bool]],
        retag_flag: bool
    ) -> None:
        for data in self.data_list:
            data.tag_insts(new_tags, lambda_funcs, retag_flag)
    
    def validate(self) -> None:
        assert len(self.data_list) == len(self.img_ids) \
            == len(self.insts_ids)
        assert len(self.cat_id_name_dict) == len(self.cat_name_id_dict)
        