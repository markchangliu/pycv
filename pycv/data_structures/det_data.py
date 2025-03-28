from dataclasses import dataclass
from typing import List

import numpy as np

from pycv.data_structures.base import BaseStructure
from pycv.data_structures.insts import Insts


@dataclass
class DetData(BaseStructure):
    img: np.ndarray
    insts: Insts

    def __post_init__(self) -> None:
        self.validate()

    def validate(self):
        if self.insts.masks is not None:
            img_h, img_w = self.img.shape[:2]
            mask_img_h, mask_img_w = self.insts.masks.img_hw
            
            if img_h != mask_img_h or img_w != mask_img_w:
                raise ValueError("img and insts.mask have different sizes")