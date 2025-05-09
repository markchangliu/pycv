from collections.abc import Sequence
from typing import List, Tuple, Union, Literal, overload

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
    - `format`: `Literal["polygon", "binary", "rle"]`
    """
    def __init__(
        self,
        data: Union[List[List[List[int]]], np.ndarray, List[dict]],
        img_hw: Tuple[int, int],
        format: Literal["polygon", "binary", "rle"]
    ) -> None:
        assert format in ["polygon", "binary", "rle"]
        assert len(img_hw) == 2
        
        if format == "polygon":
            assert isinstance(data, Sequence)

            for polys in data:
                assert isinstance(polys, list)

                for p in polys:
                    assert isinstance(p, np.ndarray)
                    assert len(p.shape) % 2 == 0
        
        elif format == "binary":
            assert isinstance(data, np.ndarray)
            assert data.shape == 3
            assert data.max().item() == 1 and data.min().item() == 0
            data = data.astype(np.uint8)
        
        elif format == "rle":
            assert isinstance(data, Sequence)

            for r in data:
                assert isinstance(r, dict)
                assert set(["size", "counts"]) == set(r.keys())
        
        self.data = data
        self.img_hw = img_hw
        self.format: Literal["polygon", "binary", "rle"] = format
    
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
        
        if self.format == "polygon" or self.format == "rle":
            new_data = []
            for i, d in enumerate(self.data):
                if i in item:
                    new_data.append(d)
        elif self.format == "binary":
            new_data = self.data[item, ...]
        
        self.data = new_data
    
    def _concat_other(self, other: "Masks") -> None:
        assert self.img_hw == other.img_hw
        
        other.convert_format(self.format)

        if self.format == "polygon" or self.format == "rle":
            new_data = self.data + other.data
        elif self.format == "binary":
            new_data = np.concat([self.data, other.data], axis=0)
        
        self.data = new_data
    
    @overload
    def concat(self, other: "Masks") -> None: ...

    @overload
    def concat(self, other: Sequence["Masks"]) -> None: ...

    def concat(
        self,
        other: Union["Masks", Sequence["Masks"]]
    ) -> None:
        if isinstance(other, "Masks"):
            other = [other]
        
        for o in other:
            self._concat_other(o)

    def convert_format(
        self, 
        dst_format: Literal["polygon", "binary", "rle"]
    ) -> None:
        if self.format == dst_format:
            return
        
        if self.format == "polygon" and dst_format == "rle":
            new_data = []

            for polys in self.data:
                rle = pycocomask.frPyObjects(polys, self.img_hw[0], self.img_hw[1])
                rle = pycocomask.merge(rle)
                new_data.append(rle)

        elif self.format == "rle" and dst_format == "binary":
            new_data = pycocomask.decode(self.data)
        
        elif self.format == 