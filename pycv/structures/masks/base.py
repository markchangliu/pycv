from collections.abc import Sequence
from typing import List, Tuple, Union, Literal, overload

import numpy as np
import pycocotools.mask as pycocomask

from pycv.structures.masks.convert import convert_masks
from pycv.structures.masks.concat import concat_masks


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