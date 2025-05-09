from collections.abc import Sequence
from typing import Literal, Union, overload

import numpy as np


class BBoxes:
    """
    Attrs:
    - `self.format`: `Literal["XYXY", "XYWH"]`,
    - `self.coords`: `np.ndarray`, `np.int_`, shape `(num, 4)`
    """

    def __init__(
        self,
        format: Literal["XYXY", "XYWH"],
        coords: np.ndarray 
    ) -> None:
        assert format == "XYXY" or format == "XYWH"
        assert isinstance(coords, np.ndarray)
        assert len(coords.shape) == 2
        assert coords.shape[1] == 4

        self.format: Literal["XYXY", "XYWH"] = format
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
        new_bboxes = BBoxes(coords, self.format)
        return new_bboxes
    
    @overload
    def concat(self, other: "BBoxes") -> None: ...

    @overload
    def concat(self, other: Sequence["BBoxes"]) -> None: ...

    def concat(
        self, 
        other: Union["BBoxes", Sequence["BBoxes"]]
    ) -> None:
        if isinstance(other, BBoxes):
            other.convert_format(self.format)
            new_coords = np.concat([self.coords, other.coords], axis=0)
        
        elif isinstance(other, Sequence):
            new_coords = []
            for b in other:
                b.convert_format(self.format)
                new_coords.append(b)
            
            new_coords = np.concat(new_coords, axis=0)
        
        else:
            raise ValueError("`other` wrong type")
        
        self.coords = new_coords

    def convert_format(
        self, 
        dst_format: Literal["XYXY", "XYWH"],
    ) -> None:
        if self.format == dst_format:
            return
        
        if self.format == "XYXY" and dst_format == "XYWH":
            ws = self.coords[:, 2] - self.coords[:, 0]
            hs = self.coords[:, 3] - self.coords[:, 1]

            self.coords[:, 2] = ws
            self.coords[:, 3] = hs
        
        if self.format == "XYWH" and dst_format == "XYWH":
            ws = self.coords[:, 2]
            hs = self.coords[:, 3]
            
            self.coords[:, 2] = self.coords[:, 0] + ws
            self.coords[:, 3] = self.coords[:, 1] + hs

