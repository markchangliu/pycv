import os
from dataclasses import dataclass
from typing import Union, List

import numpy as np

from pycv.data_structures.base import BaseStructure
from pycv.data_structures.insts import Insts


@dataclass
class DetData(BaseStructure):
    img: Union[np.ndarray, str, os.PathLike]
    insts: Insts
    img_tags: List[str]
    inst_tags: List[List[str]]

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
    
    def get_data_of_inst_tags(inst_tags: List[str]) -> "DetData":
        