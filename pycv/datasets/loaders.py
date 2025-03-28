import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from pycv.datasets.det_datasets import DetDataset
from pycv.io import load_files


def load_dataset_from_labelme_dirs(
    img_dirs: List[str, os.PathLike],
    labelme_dirs: List[str, os.PathLike],
) -> DetDataset:
    for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
        img_generator = load_files(img_dir, include_suffixes=(".png", ".jpg", ))
        
        for img_p in img_generator:
            img_dir = Path(img_p).parent
            img_stem = Path(img_p).stem
            labelme_p = os.path.join(img_dir, f"{img_stem}.json")

            if not os.path.exists(labelme_p):
                continue

