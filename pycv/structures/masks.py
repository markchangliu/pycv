from collections.abc import Sequence
from typing import List, Tuple, Union, Literal

import numpy as np
import pycocotools.mask as pycocomask


class Masks:
    """
    Attrs:
    - `self.data`: 
        - `polygons`: `List[List[List[int]]]`, shape `(num_objs, (num_polys, (num_points * 2, )))`
        - `binarys`: `np.ndarray`, `np.uint8`, shape `(num_objs, img_h, img_w)`
        - `rles`: `List[dict]`, shape `(num_objs, )`
    - `img_hw`: `Tuple[int, int]`
    - `data_format`: `Literal["polygon", "binary", "rle"]`
    """
    def __init__(
        self,
        data: Union[List[List[List[int]]], np.ndarray, List[dict]],
        img_hw: Tuple[int, int],
        data_format: Literal["polygon", "binary", "rle"]
    ) -> None:
        assert data_format in ["polygon", "binary", "rle"]
        assert len(img_hw) == 2
        
        if data_format == "polygon":
            assert isinstance(data, Sequence)

            for polys in data:
                assert isinstance(polys, list)

                for p in polys:
                    assert isinstance(p, np.ndarray)
                    assert len(p.shape) % 2 == 0
        
        elif data_format == "binary":
            assert isinstance(data, np.ndarray)
            assert data.shape == 3
            assert data.max().item() == 1 and data.min().item() == 0
            data = data.astype(np.uint8)
        
        elif data_format == "rle":
            assert isinstance(data, Sequence)

            for r in data:
                assert isinstance(r, dict)
                assert set(["size", "counts"]) == set(r.keys())
        
        self.data = data
        self.img_hw = img_hw
        self.data_format: Literal["polygon", "binary", "rle"] = data_format
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(
        self,
        item: Union[int, Sequence[int], slice, np.ndarray]
    ) -> None:
        if isinstance(item, int):
            item = [item]
        if isinstance(item, np.ndarray) and item.dtype == np.bool:
            item = np.arange(len(item))[item].tolist()
        if isinstance(item, slice):
            item = list(range(len(self.data))[item])
        
        if self.data_format == "polygon" or self.data_format == "rle":
            new_data = []
            for i, d in enumerate(self.data):
                if i in item:
                    new_data.append(d)
        elif self.data_format == "binary":
            new_data = self.data[item, ...]
        
        self.data = new_data

    def concat(
        self,
        other: Union["Masks", Sequence["Masks"]]
    ) -> None:
        if isinstance(other, "Masks"):
            other = [other]
        
        data_list = []
        
        for o in other:
            data_list.append(o.data)
        
        self.data = concat_masks(data_list)

    def convert_format(
        self, 
        dst_format: Literal["polygon", "binary", "rle"]
    ) -> None:
        new_data = convert_masks(
            self.data, self.img_hw, self.data_format, dst_format
        )
        self.data = new_data

def concat_masks(
    *masks_list: Union[List[List[List[int]]], np.ndarray, List[dict]]
) -> None:
    for m in masks_list:
        if not isinstance(m, type(masks_list[0])):
            raise ValueError("elements in `masks_list` must be same type")
    
    if isinstance(masks_list[0], list):
        new_mask = []

        for m in masks_list:
            new_mask += m
    
    elif isinstance(masks_list[0], np.ndarray):
        new_mask = np.concat(masks_list, axis=0)
    
    else:
        raise NotImplementedError
    
    return new_mask

def convert_masks(
    src_data: Union[List[List[List[float]]], np.ndarray, List[dict]],
    img_hw: Tuple[int, int],
    src_format: Literal["polygon", "binary", "rle"],
    dst_format: Literal["polygon", "binary", "rle"]
) -> Union[List[List[List[float]]], np.ndarray, List[dict]]:
    if src_format == dst_format:
        dst_data = src_data
    elif src_format == "polygon" and dst_format == "binary":
        dst_data = convert_masks_polys2binarys(src_data, img_hw)
    elif src_format == "polygon" and dst_format == "rle":
        dst_data = convert_masks_polys2rles(src_data, img_hw)
    elif src_format == "rle" and dst_format == "binary":
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