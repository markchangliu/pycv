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


def crop_img(
    img: np.ndarray,
    crop_x1y1x2y2: Tuple[int, int, int, int]
) -> np.ndarray:
    x1, y1, x2, y2 = crop_x1y1x2y2
    crop_img = img[y1:y2, x1:x2, :]
    return crop_img


def rotate_img(
    img: np.ndarray,
    angle: int,
) -> np.ndarray:
    """
    Args
    - `img`: `np.ndarray`, shape `(img_h, img_w, 3)`
    - `angle`: `int`, anti-clockwise >0, clockwise <0

    Retuns
    - `new_img`: `np.ndarray`, shape `(new_h, new_w, 3)`
    """
    img_h, img_w = img.shape[:2]
    ctr_xy = (img_w // 2, img_h // 2)
    rotate_mat = cv2.getRotationMatrix2D(ctr_xy, angle, 1)
    new_img = cv2.warpAffine(img, rotate_mat, (img_w, img_h))

    return new_img
