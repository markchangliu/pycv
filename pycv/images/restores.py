from typing import Tuple

import cv2
import numpy as np


def restore_resize_pad_img(
    new_img: np.ndarray,
    org_hw: Tuple[int, int]
) -> np.ndarray:
    """
    将等比例缩放+0值中心填充后的图片恢复。\n
    先将缩放比例较长的一边 unpad，再按照较小的缩放比例缩放图片。\n

    Args
    - `new_img`: `np.ndarray`, shape `(new_h, new_w, 3)`
    - `org_hw`: `Tuple[int, int]`, `[org_h, org_w]`

    Retuns
    - `org_img`: `np.ndarray`, shape `(org_h, org_w, 3)`
    """
    new_h, new_w = new_img.shape[:2]
    org_h, org_w = org_hw
    r = min(new_h/org_h, new_w/org_w)

    resize_h = int(org_h * r)
    resize_w = int(org_w * r)

    if resize_h == new_h:
        pad_top, pad_bottom = 0, 0
        pad_size = (new_w - resize_w) // 2
        pad_left = pad_size
        pad_right = pad_size
    elif resize_w == new_w:
        pad_left, pad_right = 0, 0
        pad_size = (new_h - resize_h) // 2
        pad_top = pad_size
        pad_bottom = pad_size
    else:
        raise NotImplementedError
    
    x1 = pad_left
    x2 = new_w - pad_right
    y1 = pad_top
    y2 = new_h - pad_bottom
    unpad_img = new_img[y1:y2, x1:x2, :]

    org_img = cv2.resize(unpad_img, (org_w, org_h))

    return org_img