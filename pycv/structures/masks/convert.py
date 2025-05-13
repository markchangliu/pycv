from typing import Union, List, Literal, Tuple

import numpy as np
import pycocotools.mask as pycocomask


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