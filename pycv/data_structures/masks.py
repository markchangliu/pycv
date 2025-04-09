from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pycocotools.mask as pycocomask


class MaskFormat(Enum):
    POLY: str = "polygon"
    BINARY: str = "binary_mat"
    RLE: str = "rle"

@dataclass
class Masks:
    data: Union[List[List[List[float]]], np.ndarray, List[dict]] # (num_objs, ...)
    img_hw: Tuple[int, int]
    format: MaskFormat

    def __post_init__(self) -> None:
        if self.format == MaskFormat.BINARY and isinstance(self.data, np.ndarray):
            self.data.astype(np.uint8)

        self.validate()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(
        self,
        item: Union[int, List[int], slice, np.ndarray]
    ) -> "Masks":
        if isinstance(item, int):
            item = [item]
        if isinstance(item, np.ndarray) and item.dtype == np.bool:
            item = np.arange(len(item))[item].tolist()
        if isinstance(item, slice):
            item = list(range(len(self.data))[item])
        
        if self.format == MaskFormat.POLY or self.format == MaskFormat.RLE:
            new_data = []
            for i, d in enumerate(self.data):
                if i in item:
                    new_data.append(d)
        elif self.format == MaskFormat.BINARY:
            new_data = self.data[item, ...]
        
        new_masks = Masks(new_data, self.img_hw, self.format)
        return new_masks

    def _concat(
        self,
        other_masks: "Masks"
    ) -> "Masks":
        assert self.img_hw == other_masks.img_hw
        
        if self.format != other_masks.format:
            other_masks.convert_format(self.format)

        if self.format == MaskFormat.POLY or self.format == MaskFormat.RLE:
            new_data = self.data + other_masks.data
        elif self.format == MaskFormat.BINARY:
            new_data = np.concat([self.data, other_masks.data], axis=0)
        
        new_masks = Masks(new_data, self.img_hw, self.format)

        return new_masks
    
    def concat(
        self, 
        other_masks: Union["Masks", List["Masks"]]
    ) -> "Masks":
        if isinstance(other_masks, Masks):
            new_masks = self._concat(other_masks)
        elif isinstance(other_masks, (list, tuple)):
            new_masks = self
            for masks in other_masks:
                new_masks = new_masks._concat(masks)
        
        return new_masks
    
    def convert_format(self, dst_format: MaskFormat) -> "Masks":
        dst_data = convert_masks(
            self.data, self.format, dst_format
        )
        dst_masks = Masks(dst_data, self.img_hw, dst_format)
        return dst_masks
    
    def validate(self):
        if self.format == MaskFormat.POLY:
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
        elif self.format == MaskFormat.BINARY:
            if not isinstance(self.mats, np.ndarray):
                raise ValueError("mats must be an array")
            if len(self.mats.shape) != 3:
                raise ValueError("mats must be of shape (num_objs, img_h, img_w)")
            if self.mats.max() != 1 or self.mats.min() != 0:
                raise ValueError("mats value must be 0/1")
        elif self.format == MaskFormat.RLE:
            if not isinstance(self.rles, list):
                raise ValueError("rles must be a list of dictionary")
            if not isinstance(self.rles[0], dict):
                raise ValueError("rles must be a list of dictionary")
            if self.rles[0].get("size", None) is None:
                raise ValueError("rle dictionary is not valid")
            if self.rles[0].get("counts", None) is None:
                raise ValueError("rle dictionary is not valid")
        else:
            raise ValueError("invalid mask format")


def convert_masks(
    src_data: Union[List[List[List[float]]], np.ndarray, List[dict]],
    img_hw: Tuple[int, int],
    src_format: MaskFormat,
    dst_format: MaskFormat
) -> Union[List[List[List[float]]], np.ndarray, List[dict]]:
    if src_format == dst_format:
        dst_data = src_data
    elif src_format == MaskFormat.POLY and dst_format == MaskFormat.BINARY:
        dst_data = convert_masks_polys2binarys(src_data, img_hw)
    elif src_format == MaskFormat.POLY and dst_format == MaskFormat.RLE:
        dst_data = convert_masks_polys2rles(src_data, img_hw)
    elif src_format == MaskFormat.RLE and dst_format == MaskFormat.BINARY:
        dst_data = convert_masks_rles2binarys(src_data)
    else:
        raise NotImplementedError
    
    return dst_data


def convert_masks_polys2rles(
    polys: List[List[List[int]]],
    img_hw: Tuple[int, int],
) -> Union[List[dict], dict]:
    """
    coco segmentation polygon转化为pycocotools rle。

    Args
    - `polys`: `List[List[List[int]]]`, 
    shape `(num_objs, (num_polys, (num_poly_points * 2, )))`, 每一个obj为
    `[[x1, y1, x2, y2, ...], [x1, y1, ...],...]`, 每一个`[x1, y1, ...]`代表一个polygon
    - `img_hw`: `Tuple[int, int]`

    Returns
    - `rles`: `List[dict]`, shape `(num_polys, )`, 每一个rle对应一个poly
    """
    rles = []

    for polys_obj in polys:
        rles_obj = pycocomask.frPyObjects(polys_obj, img_hw[0], img_hw[1])
        rles_obj = pycocomask.merge(rles_obj)
        rles.append(rles_obj)
    
    return rles


def convert_masks_rles2binarys(
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


def convert_masks_polys2binarys(
    polys: List[List[List[int]]],
    img_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Args
    - `polys`: `List[List[List[int]]]`, 
    shape `(num_objs, (num_polys, (num_poly_points * 2, )))`, 每一个obj为
    `[[x1, y1, x2, y2, ...], [x1, y1, ...],...]`, 每一个`[x1, y1, ...]`代表一个polygon
    - `img_hw`: `Tuple[int, int]`

    Returns
    - `masks`: `Array[uint8]`, 0/1, shape `(img_h, img_w, num_rles)`
    """
    rles = convert_masks_polys2rles(polys, img_hw)
    masks = convert_masks_rles2binarys(rles)
    
    return masks
