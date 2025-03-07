from typing import Union, List

import cv2
import numpy as np


class DetInsts:
    def __init__(
        self,
        scores: np.ndarray,
        cats: np.ndarray,
        bboxes: np.ndarray,
    ) -> None:
        assert len(scores) == len(cats) == len(bboxes)

        sort_idx = np.argsort(scores)[::-1]
        scores = scores[sort_idx]
        cats = cats[sort_idx]
        bboxes = bboxes[sort_idx, ...]
        
        self.scores = scores.astype(np.float32)
        self.cats = cats.astype(np.int32)
        self.bboxes = bboxes.astype(np.int32)
    
    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(
        self, 
        item: Union[int, List[int], slice, np.ndarray]
    ) -> "DetInsts":
        if isinstance(item, int):
            item = [item]
        
        scores = self.scores[item]
        cats = self.cats[item]
        bboxes = self.bboxes[item, :]

        det_objs = DetInsts(scores, cats, bboxes)

        return det_objs

class SegmInsts:
    """
    Attrs
    -----
    - `self.scores`: array, `(num_insts, )`, float32
    - `self.cats`: array, `(num_insts, )`, int32
    - `self.bboxes`: array, `(num_insts, 4)`, `x1y1x2y2`, int32
    - `self.masks`: array, `(num_insts, img_h, img_w)`, bool
    """

    def __init__(
        self,
        scores: np.ndarray,
        cats: np.ndarray,
        bboxes: np.ndarray,
        masks: np.ndarray
    ) -> None:
        assert len(scores) == len(cats) == len(bboxes) == len(masks)

        sort_idx = np.argsort(scores)[::-1]
        scores = scores[sort_idx]
        cats = cats[sort_idx]
        bboxes = bboxes[sort_idx, ...]
        masks = masks[sort_idx, ...]
        
        self.scores = scores.astype(np.float32)
        self.cats = cats.astype(np.int32)
        self.bboxes = bboxes.astype(np.int32)
        self.masks = masks.astype(np.bool_)
    
    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(
        self, 
        item: Union[int, List[int], slice, np.ndarray]
    ) -> "SegmInsts":
        if isinstance(item, int):
            item = [item]
        
        scores = self.scores[item]
        cats = self.cats[item]
        bboxes = self.bboxes[item, :]
        masks = self.masks[item, :]

        segm_insts = SegmInsts(scores, cats, bboxes, masks)

        return segm_insts


def bitmask_to_polygon(
    bitmask: np.ndarray, 
    min_cnt_area: int = 10000
) -> list:
    """
    Convert bitmask to polygon mask

    Args:
    - `bitmask`: `np.ndarray`, shape `(H, W)`, dtype `np.bool_`,
    - `min_cnt_area`: `int`
    
    Returns:
    - `polygon`: `list`, `polygon[i]` is a point `[x, y]`
    """
    bitmask = bitmask.astype(np.uint8)

    # cnts: List[array], shape (num_cnts, (num_points, 1, 2))
    cnts, _ = cv2.findContours(
        bitmask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_cnts = []

    for cnt in cnts:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < min_cnt_area:
            continue

        cnt = cv2.approxPolyDP(cnt, 15, True)
        valid_cnts.append(cnt)
    
    polygon = []

    for cnt in valid_cnts:
        cnt = np.squeeze(cnt).tolist()
        polygon += cnt
    
    return polygon

