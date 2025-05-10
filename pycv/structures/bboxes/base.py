from collections.abc import Sequence
from typing import Literal, Union

import numpy as np

from pycv.structures.bboxes.convert import convert_bboxes
from pycv.structures.bboxes.concat import concat_bboxes


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

