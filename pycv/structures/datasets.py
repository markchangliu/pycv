from typing import List, Dict, Union

import numpy as np

from pycv.structures.data import DetData


class DetDataset:
    """
    Attrs:
    - data_list: List[DetData] # (num_imgs, )
    - cat_id_name_dict: Dict[int, str] # (num_cats, )
    - cat_name_id_dict: Dict[str, int] # (num_cats, )
    - img_ids: List[int] # (num_imgs, )
    - insts_ids: List[List[int]] # (num_imgs, (num_insts_per_img, ))
    """
    def __init__(
        self,
        data_list: List[DetData],
        cat_id_name_dict: Dict[int, str],
    ) -> None:
        assert len(self.data_list) == len(self.img_ids) \
            == len(self.insts_ids)
        
        self.data_list = data_list
        self.cat_id_name_dict = cat_id_name_dict
        self.cat_name_id_dict = {v:k for k, v in cat_id_name_dict.items()}
        self.img_ids = []
        self.insts_ids = []

        self.create_index()
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def create_index(self) -> None:
        img_ids = list(range(0, len(self.data_list)))

        curr_insts_id = 0
        insts_ids = []
        for data in self.data_list:
            this_insts_ids = list(range(curr_insts_id, curr_insts_id + len(data)))
            insts_ids.append(this_insts_ids)
            curr_insts_id += len(data)
        
        self.img_ids = img_ids
        self.insts_ids = insts_ids
    
    def concat(
        self, 
        other: Union["DetDataset", List["DetDataset"]]
    ) -> "DetDataset":
        if not isinstance(other, list):
            other = [other]
        
        dataset_list = [self] + other
        new_dataset = concat_datasets(dataset_list)
        return new_dataset
    
    def update_cat_dict(
        self,
        new_cat_id_name_dict: Dict[int, str]
    ) -> None:
        cat_id_old_new_dict = {}

        for new_cat_id, cat_name in new_cat_id_name_dict.items():
            old_cat_id = self.cat_name_id_dict[cat_name]
            cat_id_old_new_dict[old_cat_id] = new_cat_id
        
        for data in self.data_list:
            insts = data.insts
            old_cat_ids = insts.cat_ids.tolist()
            new_cat_ids = [cat_id_old_new_dict[k] for k in old_cat_ids]
            new_cat_ids = np.asarray(new_cat_ids, dtype=np.int_)
            insts.cat_ids = new_cat_ids


def concat_datasets(
    *dataset_list: "DetDataset",
) -> "DetDataset":
    new_data_list = []
    new_cat_id_name_dict = {}

    curr_cat_id = 0
    for dataset in dataset_list:
        new_data_list += dataset.data_list

        for k, v in dataset.cat_id_name_dict.items():
            if v in new_cat_id_name_dict.values():
                continue

            new_cat_id_name_dict[curr_cat_id] = v
            curr_cat_id += 1

    new_dataset = DetDataset(new_data_list, new_cat_id_name_dict)
    new_dataset.update_cat_dict(new_cat_id_name_dict)

    return new_dataset