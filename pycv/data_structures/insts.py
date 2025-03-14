from typing import Union, List, Optional

import numpy as np


class Insts:
    """
    Attrs
    -----
    - `self.scores`: `Array[float_]`, `(num_insts, )`
    - `self.cats`: `Array[int_]`, `(num_insts, )`
    - `self.bboxes`: `Array[int_]`, `(num_insts, 4)`, `x1y1x2y2`
    - `self.masks`: `Optional[Array[uint8]]`, `(num_insts, img_h, img_w)`, 0/1
    """

    def __init__(
        self,
        scores: np.ndarray,
        cats: np.ndarray,
        bboxes: np.ndarray,
        masks: Optional[np.ndarray]
    ) -> None:
        if masks is not None:
            assert len(scores) == len(cats) == len(bboxes) == len(masks)
        else:
            assert len(scores) == len(cats) == len(bboxes)

        sort_idx = np.argsort(scores)[::-1]
        scores = scores[sort_idx]
        cats = cats[sort_idx]
        bboxes = bboxes[sort_idx, ...]
        
        self.scores = scores.astype(np.float_)
        self.cats = cats.astype(np.int_)
        self.bboxes = bboxes.astype(np.int_)

        if masks is not None:
            masks = masks[sort_idx, ...]
            self.masks = masks.astype(np.bool_)
        else:
            self.masks = None
    
    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(
        self, 
        item: Union[int, List[int], slice, np.ndarray]
    ) -> "Insts":
        if isinstance(item, int):
            item = [item]
        
        scores = self.scores[item]
        cats = self.cats[item]
        bboxes = self.bboxes[item, :]

        if self.masks is not None:
            masks = self.masks[item, :]
            insts = Insts(scores, cats, bboxes, masks)
        else:
            insts = Insts(scores, cats, bboxes, None)

        return insts