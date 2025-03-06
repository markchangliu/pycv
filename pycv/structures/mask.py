from typing import List, Tuple, Union

import numpy as np
import pycocotools.mask as pycocomask


def polyscoco2rles(
    polys: List[List[int]],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> Union[List[dict], dict]:
    """
    coco segmentation polygon转化为pycocotools rle。

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
    masks = pycocomask.decode(rles)
    return masks


def polyscoco2masks(
    polys: List[List[int]],
    img_hw: Tuple[int, int],
    merge_flag: bool
) -> np.ndarray:
    """
    Args
    - `polys`: `List[List[int]]`, shape `(num_polys, (num_poly_points * 2, ))`,
    `[[x1, y1, x2, y2, ...], [x1, y1, ...],...]`, 每一个`[x1, y1, ...]`代表一个polygon
    - `img_hw`: `Tuple[int, int]`
    - `merge_flag`: `bool`, 是否融合为单一rle

    Returns
    - `masks_or_mask`: `Array[uint8]`
        - `masks`: 若`merge_flag=False`, shape `(num_polys, img_h, img_w)`,
        每一个rle对应一个poly
        - `mask`: 若`merge_flag=True`, shape `(1, img_h, img_w)`
    """
    rles = polyscoco2rles(polys, img_hw, merge_flag)
    rles = [rles] if merge_flag else rles
    masks = rles2masks(rles)
    return masks


def center_pad_masks(
    masks: np.ndarray,
    new_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Args
    - `masks`: `Array[uint8]`, shape `(num_masks, img_h, img_w)`
    - `new_hw`: Tuple[int, int],

    Return
    - `new_masks`: `Array[uint8]`, shape `(num_masks, new_h, new_w)`
    """
    num_masks, img_h, img_w = masks.shape
    new_h, new_w = new_hw

    assert new_h >= img_h and new_w >= img_w

    pad_l = (new_w - img_w // 2)
    pad_r = pad_l + img_w
    pad_t = (new_h - img_w) // 2
    pad_b = pad_t + img_h

    new_masks = np.zeros((num_masks, new_h, new_w), dtype=np.uint8)
    new_masks[:, pad_t:pad_b, pad_l:pad_r] = masks

    return new_masks