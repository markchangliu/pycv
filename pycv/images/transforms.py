from typing import Tuple

import cv2
import numpy as np


def resize_pad_img(
    img: np.ndarray,
    new_hw: Tuple[int, int]
) -> np.ndarray:
    """
    等比例缩放+中心0值填充，不改变宽高比。\n
    先将图片按照较小缩放比缩放，再对未达到目标尺寸的一边进行0值中心填充。\n

    Args
    - `img`: `np.ndarray`, shape `(img_h, img_w, 3)`
    - `new_hw`: `Tuple[int, int]`, `[new_h, new_w]`

    Retuns
    - `new_img`: `np.ndarray`, shape `(new_h, new_w, 3)`
    """
    img_h, img_w = img.shape[:2]
    new_h, new_w = new_hw
    r = min(new_h/img_h, new_w/img_w)

    resize_h = int(img_h * r)
    resize_w = int(img_w * r)

    if resize_h == new_h:
        pad_top, pad_bottom = 0, 0
        pad_size = (new_w - resize_w) // 2
        pad_left = pad_size
        pad_right = pad_size
    elif resize_w == new_w:
        pad_size = (new_h - resize_h) // 2
        pad_top = pad_size
        pad_bottom = pad_size
        pad_left, pad_right = 0, 0
    else:
        raise NotImplementedError
    
    resize_img = cv2.resize(img, (resize_w, resize_h))
    new_img = cv2.copyMakeBorder(
        resize_img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, None, value = 0
    )

    return new_img