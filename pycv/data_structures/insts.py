from typing import Union, List, Optional

import numpy as np

from pycv.data_structures.bboxes import BBoxFormat, BBoxes
from pycv.data_structures.masks import MaskFormat, Masks


class Insts:
    """
    Attrs
    -----
    - `self.confs`: `Array[float]`, `(num_insts, )`
    - `self.cat_ids`: `Array[int]`, `(num_insts, )`
    - `self.bboxes`: `BBoxes`, `(num_insts, 4)`,
    - `self.masks`: `Masks`, `(num_insts, ...)`,
    """

    def __init__(
        self,
        confs: np.ndarray,
        cat_ids: np.ndarray,
        bboxes: BBoxes,
        masks: Optional[Masks]
    ) -> None:
        if masks is not None:
            assert len(confs) == len(cat_ids) == len(bboxes) == len(masks)
            bboxes.convert_format(BBoxFormat.XYWH)
            masks.convert_format(MaskFormat.BINARY)
        else:
            assert len(confs) == len(cat_ids) == len(bboxes)
            bboxes.convert_format(BBoxFormat.XYWH)
        
        sort_idx = np.argsort(confs)[::-1]
        confs = confs[sort_idx]
        cat_ids = cat_ids[sort_idx]
        bboxes = bboxes[sort_idx]

        if masks is not None:
            mask = masks[sort_idx]
        
        self.confs = confs.astype(np.float_)
        self.cat_ids = cat_ids.astype(np.int_)
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