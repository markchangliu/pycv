from dataclasses import dataclass
from typing import List, Dict, Set, Union

import numpy as np

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

    def validate(self) -> None:
        assert len(self.data_list) == len(self.img_ids) \
            == len(self.insts_ids)
        assert len(self.cat_id_name_dict) == len(self.cat_name_id_dict)

    def get_data_of_cat_ids(
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
            new_data = data.get_insts_of_cat_ids(target_cat_ids)
            
            if len(new_data) == 0:
                continue

            new_data_list.append(data)
        

        