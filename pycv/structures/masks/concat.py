from typing import Union, List

import numpy as np


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