from typing import List, Tuple, Union

import numpy as np
import pycocotools.mask as pycocomask


def polysLabelme2rles(
    polys: List[List[int]],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> Union[List[dict], dict]:
    """
    将labelme segmentation polygon转化为pycocotools mask rle。

    Args
    - `polys`: `List[List[int]]`, shape `(num_polys, (num_poly_points * 2, ))`,
    `[[x1, y1, x2, y2, ...], [x1, y1, ...],...]`, 每一个`[x1, y1, ...]`代表一个polygon
    - `img_hw`: `Tuple[int, int]`
    - `merge_flag`: `bool`, 是否融合为单一rle

    Returns
    - `rles_or_rle`: `Union[List[dict], dict]`
        - `rles`: 若`merge_flag=False`, 则返回`List[dict]`, shape `(num_polys, )`,
        每一个rle对应一个poly
        - `rle`: 若`merge_flag=True`, 则返回`dict`, 是将所有poly融合后的rle
    """
    rles = pycocomask.frPyObjects(polys, img_hw[0], img_hw[1])

    if not merge_flag:
        return rles
    else:
        rle = pycocomask.merge(rles)
        return rle


def rles2masks(
    rles: List[dict]
) -> np.ndarray:
    """
    将rles转化为masks。

    Args
    - `rles`: `List[dict]`, shape `(num_rles, )`

    Returns
    - `masks`: `Array[uint8]`, 0/1, shape `(img_h, img_w, num_rles)`
    """
    masks = pycocomask.decode()