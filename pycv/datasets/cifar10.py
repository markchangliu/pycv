import os
import pickle
import multiprocessing as mp
from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np


CIFAR10_CATEGORY_ID_NAME_DICT = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

CIFAR10_CATEGORY_NAME_ID_DICT = {
    v: k for k, v in CIFAR10_CATEGORY_ID_NAME_DICT.items()
}


def unpickle_batch_f(
    batch_fp: Union[os.PathLike, str]
) -> dict:
    """
    Returns
    - `category_ids`: `List[int]`, shape `(num_data, )`
    - `imgs_flat`: `np.ndarray`, shape `(num_data, 3072=32*32*3)`
    - `img_names`: `List[str]`, shape `(num_data, )`
    """
    with open(batch_fp, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    
    category_ids = data_dict[b"labels"]
    imgs_flat = data_dict[b"data"]
    img_names = data_dict[b"filenames"]

    return category_ids, imgs_flat, img_names

def img_flat2bgr(
    img_flat: np.ndarray
) -> np.ndarray:
    """
    Args
    - `img_flat`: `np.ndarray`, shape `(3072=32*32*3)`

    Returns
    - `img_bgr`: `np.ndarray`, shape `(32, 32, 3)`
    """
    r_channel = img_flat[:1024].reshape((32, 32))
    g_channel = img_flat[1024:2048].reshape((32, 32))
    b_channel = img_flat[2048:].reshape((32, 32))

    img_bgr = np.stack([b_channel, g_channel, r_channel], axis=2)

    return img_bgr

def dump_batch_f(
    batch_fp: Union[os.PathLike, str],
    dump_dir: Union[os.PathLike, str]
) -> None:
    """
    `dump_dir`: directory tree
    - `{category_name}: {png_img}`
    """
    category_ids, imgs_flat, img_names = unpickle_batch_f(batch_fp)

    img_id = 0

    for category_id, img_flat, img_name in zip(category_ids, imgs_flat, img_names):
        img_name = img_name.decode("UTF-8")
        category = CIFAR10_CATEGORY_ID_NAME_DICT[category_id]
        img_folder = os.path.join(dump_dir, category)
        img_p = os.path.join(img_folder, img_name)

        img_bgr = img_flat2bgr(img_flat)

        os.makedirs(img_folder, exist_ok=True)
        cv2.imwrite(img_p, img_bgr)

        img_id += 1
        
        if img_id % 500 == 0:
            print(f"dumped {img_id} images")

def dump_batch_fs(
    batch_root: Union[os.PathLike, str],
    dump_root: Union[os.PathLike, str]
) -> None:
    """
    Directory tree
    - `batch_root`:
        - `data_batch_[1~5]`
    - `dump_root`:
        - `train/test`:
            - `{category_name}: {png_img}`
    """
    num_procs = min(6, os.cpu_count() - 1)
    filenames = os.listdir(batch_root)
    filenames.sort()
    args = []

    for filename in filenames:
        if "data_batch" in filename:
            batch_f = os.path.join(batch_root, filename)
            dump_dir = os.path.join(dump_root, "train")
        elif "test_batch" in filename:
            batch_f = os.path.join(batch_root, filename)
            dump_dir = os.path.join(dump_root, "test")
        else:
            continue

        args.append((batch_f, dump_dir))

    with mp.Pool(num_procs) as pool:
        res = pool.starmap_async(dump_batch_f, args)

        for msg in res.get():
            print(msg)

