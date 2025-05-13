from typing import Tuple, Set, Sequence

import numpy as np


class Image:
    """
    Attrs:
    - `mat`: `np.ndarray`, `np.uint8`, shape `(img_h, img_w, [channel])`
    - `img_hw`: `Tuple[int, int]`
    - `org_hw`: `Tuple[int, int]`
    - `tags`: `Set[str]`
    - `id`: `int`

    """
    def __init__(
        self,
        mat: np.ndarray,
        img_hw: Tuple[int, int],
        org_hw: Tuple[int, int],
        tags: Set[str],
        id: int
    ) -> None:
        self.mat = mat
        self.img_hw = img_hw
        self.org_hw = org_hw
        self.tags = tags
        self.id = id
    
    def retrieve_by_tags(
        self,
        target_tags: Sequence[str]
    ) -> bool:
        target_tags = set(target_tags)
        retrieve_flag =  len(self.tags.intersection(target_tags)) > 0
        return retrieve_flag
    
    def retrieve_by_id(
        self,
        target_ids: Sequence[int]
    ) -> bool:
        target_ids = set(target_ids)
        retrieve_flag = self.id in target_ids
        return retrieve_flag