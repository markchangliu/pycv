import os
from typing import Tuple, Union

import numpy as np


class Image:
    """
    Attrs:
    - `mat`: `np.ndarray`, `np.uint8`, shape `(img_h, img_w, [channel])`
    - `img_hw`: `Tuple[int, int]`
    - `org_hw`: `Tuple[int, int]`
    - `img_p`: `Union[str, os.PathLike]`
    """
    def __init__(
        self,
        mat: np.ndarray,
        img_hw: Tuple[int, int],
        org_hw: Tuple[int, int],
        img_p: Union[str, os.PathLike]
    ) -> None:
        self.mat = mat
        self.img_hw = img_hw
        self.org_hw = org_hw
        self.img_p = img_p