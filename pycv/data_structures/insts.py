from enum import Enum
from dataclasses import dataclass
from typing import Union, List, Optional

import numpy as np

from pycv.data_structures.base import BaseStructure
from pycv.data_structures.bboxes import BBoxFormat, BBoxes
from pycv.data_structures.masks import MaskFormat, Masks


class InstsType(Enum):
    GT: str = "ground_truth"
    DT: str = "detection"


@dataclass
class Insts(BaseStructure):
    """
    Attrs
    -----
    - `self.confs`: `Array[float]`, `(num_insts, )`
    - `self.cat_ids`: `Array[int]`, `(num_insts, )`
    - `self.bboxes`: `BBoxes`, `(num_insts, 4)`,
    - `self.masks`: `Masks`, `(num_insts, ...)`,
    """
    type: InstsType
    confs: np.ndarray
    cat_ids: np.ndarray
    bboxes: BBoxes
    masks: Optional[Masks]

    def __post_init__(self) -> None:
        self.validate()
    
    def __len__(self) -> int:
        return len(self.confs)

    def __getitem__(
        self, 
        item: Union[int, List[int], slice, np.ndarray]
    ) -> "Insts":
        if isinstance(item, int):
            item = [item]
        
        confs = self.confs[item]
        cat_ids = self.cat_ids[item]
        bboxes = self.bboxes[item]

        if self.masks is not None:
            masks = self.masks[item]
            insts = Insts(confs, cat_ids, bboxes, masks)
        else:
            insts = Insts(confs, cat_ids, bboxes, None)

        return insts

    def validate(self):
        if self.masks is not None:
            assert len(self.confs) == len(self.cat_ids) \
                == len(self.bboxes) == len(self.masks)
            self.bboxes.convert_format(BBoxFormat.XYWH)
            self.masks.convert_format(MaskFormat.BINARY)
        else:
            assert len(self.confs) == len(self.cat_ids) == len(self.bboxes)
            self.bboxes.convert_format(BBoxFormat.XYWH)