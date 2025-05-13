from collections.abc import Sequence
from typing import Literal, Union

import numpy as np


class BBoxes:
    """
    Attrs:
    - `self.data_format`: `Literal["XYXY", "XYWH"]`,
    - `self.coords`: `np.ndarray`, `np.int_`, shape `(num, 4)`
    """

    def __init__(
        self,
        data_format: Literal["XYXY", "XYWH"],
        coords: np.ndarray 
    ) -> None:
        assert data_format == "XYXY" or data_format == "XYWH"
        assert isinstance(coords, np.ndarray)
        assert len(coords.shape) == 2
        assert coords.shape[1] == 4

        self.data_format: Literal["XYXY", "XYWH"] = data_format
        self.coords = coords.astype(np.int_)
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(
        self,
        item: Union[int, Sequence[int], slice, np.ndarray]
    ) -> "BBoxes":
        if isinstance(item, int):
            item = [item]
        
        coords = self.coords[item, :]
        new_bboxes = BBoxes(coords, self.data_format)
        return new_bboxes

    def concat(
        self, 
        other: Union["BBoxes", Sequence["BBoxes"]]
    ) -> None:
        if isinstance(other, BBoxes):
            other = [other]
        
        coords_list = [self.coords]

        for b in other:
            b.convert_format(self.data_format)
            coords_list.append(b.coords)
        
        new_coords = concat_bboxes(*coords_list)
        
        self.coords = new_coords

    def convert_format(
        self, 
        dst_format: Literal["XYXY", "XYWH"],
    ) -> None:
        self.coords = convert_bboxes(self.coords, self.data_format, dst_format)


def concat_bboxes(*coords_list: np.ndarray) -> np.ndarray:
    new_coords = np.concatenate(coords_list, axis=0)
    return new_coords

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