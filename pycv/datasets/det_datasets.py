from dataclasses import dataclass
from typing import List, Dict

from pycv.data_structures.det_data import DetData


@dataclass
class DetDataset:
    data: List[DetData] # (num_imgs, )
    cat_id_name_dict: Dict[int, str] # (num_cats, )
    img_ids: List[int] # (num_imgs, )
    inst_ids: List[List[int]] # (num_imgs, (num_insts_per_img, ))
