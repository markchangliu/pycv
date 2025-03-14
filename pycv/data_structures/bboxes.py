import copy
from enum import Enum
from dataclasses import dataclass

import numpy as np

from pycv.data_structures.base import BaseStructure, DataType
from pycv.data_structures.converters import convert_bboxes


class BBoxFormat(Enum):
    XYXY: str = "xmin, ymin, xmax, ymax"
    XYWH: str = "x_center, y_center, width, height"


@dataclass
class BBoxes(BaseStructure):
    coords: np.ndarray # shape (num_objs, 4)
    format: BBoxFormat
    confidence: float
    class_id: int

    def __post_init__(self) -> None:
        self.data_type = DataType.BBOXES
    
    def validate(self) -> None:
        if len(self.coords.shape) != 2 or self.coords.shape[-1] != 4:
            raise ValueError("BBoxes coords must be of shape (num, 4)")

        if self.format == BBoxFormat.XYXY:
            if np.sum(self.coords[:, 0] >= self.coords[:, 1]) > 0:
                raise ValueError("xmax must be greater than xmin")
            if np.sum(self.coords[:, 1] >= self.coords[:, 2]) > 0:
                raise ValueError("ymax must be greater than ymin")

    def convert_format(self, dst_format: BBoxFormat) -> "BBoxes":
        dst_coords = convert_bboxes(
            self.coords, self.format, dst_format
        )
        dst_bboxes = BBoxes(dst_coords, dst_format, self.confidence, self.class_id)
        return dst_bboxes
