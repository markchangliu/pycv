from enum import Enum
from dataclasses import dataclass
from typing import Union, List, Optional

import numpy as np

from pycv.data_structures.bboxes import BBoxFormat, BBoxes
from pycv.data_structures.masks import MaskFormat, Masks


@dataclass
class Insts:
    """
    Attrs
    -----
    - `self.confs`: `Array[float]`, `(num_insts, )`
    - `self.cat_ids`: `Array[int]`, `(num_insts, )`
    - `self.bboxes`: `BBoxes`, `(num_insts, 4)`,
    - `self.masks`: `Masks`, `(num_insts, ...)`,
    """
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
    
    def concat(
        self, 
        other_insts: Union["Insts", List["Insts"]]
    ) -> "Insts":
        if isinstance(other_insts, Insts):
            new_confs = np.concat([self.confs, other_insts.confs])
            new_cat_ids = np.concat([self.cat_ids, other_insts.cat_ids])
        elif isinstance(other_insts, (list, tuple)):
            new_confs = [self.confs] + [i.confs for i in other_insts]
            new_confs = np.concat(new_confs, axis=0)
            new_cat_ids = [self.cat_ids] + [i.cat_ids for i in other_insts]
            new_cat_ids = np.concat(new_cat_ids, axis=0)

        new_bboxes = self.bboxes.concat(other_insts.bboxes)
        new_masks = self.masks.concat(other_insts.masks)
        new_insts = Insts(new_confs, new_cat_ids, new_bboxes, new_masks)
        return new_insts

    def validate(self):
        if self.masks is not None:
            assert len(self.confs) == len(self.cat_ids) \
                == len(self.bboxes) == len(self.masks)
            self.bboxes.convert_format(BBoxFormat.XYWH)
            self.masks.convert_format(MaskFormat.BINARY)
        else:
            assert len(self.confs) == len(self.cat_ids) \
                == len(self.bboxes)
            self.bboxes.convert_format(BBoxFormat.XYWH)