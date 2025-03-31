from dataclasses import dataclass
from typing import List, Dict

from pycv.data_structures.det_data import DetData


@dataclass
class DetDataset:
    data_list: List[DetData] # (num_imgs, )
    cat_id_name_dict: Dict[int, str] # (num_cats, )
    cat_name_id_dict: Dict[str, int] # (num_cats, )
    img_ids: List[int] # (num_imgs, )
    inst_ids: List[List[int]] # (num_imgs, (num_insts_per_img, ))

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        assert len(self.data_list) == len(self.img_ids) \
            == len(self.inst_ids)
        assert len(self.cat_id_name_dict) == len(self.cat_name_id_dict)

    def get_data_of_cat_ids(self, cat_ids: List[int]) -> "DetDataset":
        new_cat_id_name_dict = {}
        new_cat_name_id_dict = {}

        for i, n in self.cat_id_name_dict.items():
            if i in cat_ids:
                new_cat_id_name_dict[i] = n
                new_cat_name_id_dict[n] = i
        
        