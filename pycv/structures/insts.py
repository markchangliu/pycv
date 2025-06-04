from typing import Union, List, Optional

import numpy as np

from pycv.structures.bboxes import BBoxes
from pycv.structures.masks import Masks


class Insts:
    """
    Attrs
    -----
    - `confs`: `np.ndarray`, `np.float32`, `(num_insts, )`
    - `cat_ids`: `np.ndarray`, `np.int32`, `(num_insts, )`
    - `bboxes`: `BBoxes`, `(num_insts, 4)`, `XYXY`
    - `masks`: `Optional[Masks]`, `(num_insts, ...)`, `binary`

    Methods
    -----
    - `concat`
    """

    def __init__(
        self,
        confs: np.ndarray,
        cat_ids: np.ndarray,
        bboxes: BBoxes,
        masks: Optional[Masks],
    ) -> None:
        assert len(confs) == len(cat_ids) == len(bboxes)
        assert len(confs.shape) == 1 and len(cat_ids.shape) == 1
        
        if masks is not None:
            assert len(confs) == len(masks)
            bboxes.convert_format("XYXY")
            masks.convert_format("binary")
        else:
            bboxes.convert_format("XYXY")

        self.confs = confs.astype(np.float32)
        self.cat_ids = cat_ids.astype(np.int32)
        self.bboxes = bboxes
        self.masks = masks
    
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
    ) -> None:
        if isinstance(other_insts, Insts):
            other_insts = [other_insts]
        
        new_confs = [self.confs]
        new_cat_ids = [self.cat_ids]
        other_bboxes = []
        other_masks = []

        for i in other_insts:
            new_confs += i.confs
            new_cat_ids += i.cat_ids
            other_bboxes += i.bboxes
            
            if self.masks is not None:
                new_masks += i.masks

        self.confs = np.concat(new_confs, axis = 0)
        self.cat_ids = np.concat(new_cat_ids, axis = 0)
        self.bboxes = self.bboxes.concat(other_bboxes)
        
        if self.masks is not None:
            self.masks = self.masks.concat(other_masks)
        
        