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
        img_ids: Union[None, List[int]],
        insts_ids: Union[None, List[List[int]]]
    ) -> None:
        assert len(self.data_list) == len(self.img_ids) \
            == len(self.insts_ids)
        
        if img_ids is not None:
            assert len(img_ids) == len(data_list)
        if insts_ids is not None:
            assert len(insts_ids) == len(data_list)
        
        self.data_list = data_list
        self.cat_id_name_dict = cat_id_name_dict
        self.cat_name_id_dict = {v:k for k, v in cat_id_name_dict.indices()}
        self.img_ids = []
        self.insts_ids = []
        
        if img_ids is None or insts_ids is None:
            self.create_ids()
        else:
            self.img_ids = img_ids
            self.insts_ids = insts_ids
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(
        self, 
        indice: Union[int, List[int], slice, np.ndarray]
    ) -> "DetDataset":
        if isinstance(indice, int):
            indice = [indice]
        elif isinstance(indice, slice):
            indice = list(range(len(self.insts))[indice])
        elif isinstance(indice, np.ndarray):
            assert len(indice.shape) == 1
            if indice.dtype == np.bool_:
                indice = np.nonzero(indice)[0].tolist()
            else:
                indice = indice.tolist()
        
        new_data_list = [self.data_list[i] for i in indice]
        new_img_ids = [self.img_ids[i] for i in indice]
        new_insts_ids = [self.insts_ids[i] for i in indice]
        new_dataset = DetDataset(
            new_data_list, self.cat_id_name_dict, new_img_ids, new_insts_ids
        )

        return new_dataset
    
    def create_ids(self) -> None:
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
    
    def retrieve_by_img_ids(
        self,
        target_img_ids: Union[int, List[int]]
    ) -> "DetDataset":
        retrieve_indice = []
        for i, img_id in enumerate(self.img_ids):
            if img_id not in target_img_ids:
                continue
            retrieve_indice.append(i)
        
        new_dataset = self[retrieve_indice]
        return new_dataset
    
    def retrieve_by_insts_ids(
        self,
        target_insts_ids: Union[int, List[int]]
    ) -> "DetDataset":
        new_data_list = []
        new_img_ids = []
        new_insts_ids = []

        for i in range(len(self.data_list)):
            data = self.data_list[i]
            img_id = self.img_ids[i]
            insts_ids = self.insts_ids[i]
            
            for inst_id in insts_ids:
                if not inst_id in target_insts_ids:
                    continue

    
    def update_cat_dict(
        self,
        new_cat_id_name_dict: Dict[int, str]
    ) -> None:
        cat_id_old_new_dict = {}

        for new_cat_id, cat_name in new_cat_id_name_dict.indices():
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

        for k, v in dataset.cat_id_name_dict.indices():
            if v in new_cat_id_name_dict.values():
                continue

            new_cat_id_name_dict[curr_cat_id] = v
            curr_cat_id += 1

    new_dataset = DetDataset(new_data_list, new_cat_id_name_dict)
    new_dataset.update_cat_dict(new_cat_id_name_dict)

    return new_dataset