from typing import Literal

import numpy as np


def convert_bboxes(
    coords: np.ndarray,
    src_format: Literal["XYXY", "XYWH"],
    dst_format: Literal["XYXY", "XYWH"],
) -> np.ndarray:
    if src_format == dst_format:
        return coords
    elif src_format == "XYWH" and dst_format == "XYXY":
        new_coords = convert_bboxes_xywh2xyxy(coords)
    elif src_format == "XYXY" and dst_format == "XYWH":
        new_coords = convert_bboxes_xyxy2xywh
    else:
        raise NotImplementedError
    
    return new_coords


def convert_bboxes_xywh2xyxy(
    coords_xywh: np.ndarray
) -> np.ndarray:
    """
    Args
    - `coords_xywh`: `Array[int]`, shape `(num_bbox, 4)`

    Returns
    - `coords_xywh`: `Array[int]`, shape `(num_bbox, 4)`
    """
    ws = coords_xywh[:, 2]
    hs = coords_xywh[:, 3]
    
    coords_xyxy = coords_xywh
    coords_xyxy[:, 2] = coords_xyxy[:, 0] + ws
    coords_xyxy[:, 3] = coords_xyxy[:, 1] + hs

    return coords_xyxy


def convert_bboxes_xyxy2xywh(
    coords_xyxy: np.ndarray
) -> np.ndarray:
    """
    Args
    - `bboxes_xyxy`: `Array[int]`, shape `(num_bbox, 4)`

    Returns
    - `bboxes_xywh`: `Array[int]`, shape `(num_bbox, 4)`
    """
    ws = coords_xyxy[:, 2] - coords_xyxy[:, 0]
    hs = coords_xyxy[:, 3] - coords_xyxy[:, 1]

    coords_xywh = coords_xyxy
    coords_xywh[:, 2] = ws
    coords_xywh[:, 3] = hs

    return coords_xywh