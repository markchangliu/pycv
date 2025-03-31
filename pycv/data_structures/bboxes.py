import copy
from enum import Enum
from dataclasses import dataclass
from typing import Union, List

import numpy as np

from pycv.data_structures.base import BaseStructure, DataType


class BBoxFormat(Enum):
    XYXY: str = "xmin, ymin, xmax, ymax"
    XYWH: str = "x_center, y_center, width, height"


@dataclass
class BBoxes(BaseStructure):
    coords: np.ndarray # shape (num_objs, 4)
    format: BBoxFormat

    def __post_init__(self) -> None:
        self.data_type = DataType.BBOXES
        self.coords.astype(np.int_)
        self.validate()
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(
        self,
        item: Union[int, List[int], slice, np.ndarray]
    ) -> "BBoxes":
        if isinstance(item, int):
            item = [item]
        
        coords = self.coords[item, :]
        new_bboxes = BBoxes(coords, self.format)
        return new_bboxes
    
    def _concat(self, other_bboxes: "BBoxes") -> "BBoxes":
        if self.format != other_bboxes.format:
            other_bboxes.convert_format(self.format)
        
        new_coords = np.concat([self.coords, other_bboxes.coords], axis=0)
        new_bboxes = BBoxes(new_coords, self.format)

        return new_bboxes
    
    def concat(
        self,
        other_bboxes: Union["BBoxes", List["BBoxes"]]
    ) -> "BBoxes":
        new_bboxes = self
        for bboxes in other_bboxes:
            new_bboxes = new_bboxes._cocnat(bboxes)

        return new_bboxes

    def convert_format(self, dst_format: BBoxFormat) -> "BBoxes":
        dst_coords = convert_bboxes(
            self.coords, self.format, dst_format
        )
        dst_bboxes = BBoxes(dst_coords, dst_format, self.confidence, self.class_id)
        return dst_bboxes
    
    def validate(self) -> None:
        if len(self.coords.shape) != 2 or self.coords.shape[-1] != 4:
            raise ValueError("BBoxes coords must be of shape (num, 4)")

        if self.format == BBoxFormat.XYXY:
            if np.sum(self.coords[:, 0] >= self.coords[:, 1]) > 0:
                raise ValueError("xmax must be greater than xmin")
            if np.sum(self.coords[:, 1] >= self.coords[:, 2]) > 0:
                raise ValueError("ymax must be greater than ymin")


def convert_bboxes(
    src_coords: np.ndarray,
    src_format: BBoxFormat,
    dst_format: BBoxFormat
) -> np.ndarray:
    if src_format == dst_format:
        dst_coords = src_coords
    elif src_format == BBoxFormat.XYXY and dst_format == BBoxFormat.XYWH:
        dst_coords = convert_bboxes_xyxy2xywh(src_coords)
    elif src_format == BBoxFormat.XYWH and dst_format == BBoxFormat.XYXY:
        dst_coords = convert_bboxes_xywh2xyxy(src_coords)
    else:
        raise NotImplementedError

    return dst_coords


def convert_bboxes_xywh2xyxy(
    bboxes_xywh: np.ndarray
) -> np.ndarray:
    """
    Args
    - `bboxes_xywh`: `Array[int]`, shape `(num_bbox, 4)`

    Returns
    - `bboxes_xyxy`: `Array[int]`, shape `(num_bbox, 4)`
    """
    ws = bboxes_xywh[:, 2]
    hs = bboxes_xywh[:, 3]
    
    bboxes_xyxy = copy.deepcopy(bboxes_xywh)
    bboxes_xyxy[:, 2] = bboxes_xywh[:, 0] + ws
    bboxes_xyxy[:, 3] = bboxes_xywh[:, 1] + hs

    return bboxes_xyxy


def convert_bboxes_xyxy2xywh(
    bboxes_xyxy: np.ndarray
) -> np.ndarray:
    """
    Args
    - `bboxes_xyxy`: `Array[int]`, shape `(num_bbox, 4)`

    Returns
    - `bboxes_xywh`: `Array[int]`, shape `(num_bbox, 4)`
    """
    ws = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
    hs = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]

    bboxes_xywh = copy.deepcopy(bboxes_xyxy)
    bboxes_xywh[:, 2] = ws
    bboxes_xywh[:, 3] = hs

    return bboxes_xywh
