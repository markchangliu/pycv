from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pycocotools.mask as pycocomask

from pycv.data_structures.base import DataType, BaseStructure


class MaskFormat(Enum):
    POLY: str = "polygon"
    BINARY: str = "binary"
    RLE: str = "rle"


@dataclass
class PolyMasks(BaseStructure):
    polys: List[List[List[float]]] # shape (num_objs, (num_polys, (num_points * 2, )))
    confidence: float
    class_id: int
    img_hw: Tuple[int, int]

    def __post_init__(self) -> None:
        self.data_type = DataType.MASKS
        self.format = MaskFormat.POLY
    
    def validate(self) -> None:
        error_msg = (
            "polys must be a 3-nested list, representing"
            " shape (num_objs, (num_polys, (num_points * 2, )))"
        )

        if not isinstance(self.polys, list):
            raise ValueError(error_msg)
        if not isinstance(self.polys[0], list):
            raise ValueError(error_msg)
        if len(self.polys[0][0]) // 2 == 1:
            raise ValueError(error_msg)


@dataclass
class BinaryMasks(BaseStructure):
    mats: np.ndarray # shape (num_objs, img_h, img_w)
    confidence: float
    class_id: int

    def __post_init__(self) -> None:
        self.data_type = DataType.MASKS
        self.format = MaskFormat.BINARY
        self.img_hw = [self.mats[0].shape[0], self.mats[0].shape[1]]
    
    def validate(self) -> None:
        if not isinstance(self.mats, np.ndarray):
            raise ValueError("mats must be an array")
        if len(self.mats.shape) != 3:
            raise ValueError("mats must be of shape (num_objs, img_h, img_w)")
        if self.mats.max() != 1 or self.mats.min() != 0:
            raise ValueError("mats value must be 0/1")


@dataclass
class RleMasks(BaseStructure):
    rles: List[dict]
    confidence: float
    class_id: int

    def __post_init__(self) -> None:
        self.data_type = DataType.MASKS
        self.format = MaskFormat.RLE
        self.img_hw = self.rles[0]["size"]
    
    def validate(self) -> None:
        if not isinstance(self.rles, list):
            raise ValueError("rles must be a list of dictionary")
        if not 



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


def restore_center_pad_masks(
    
):
    pass