import numpy as np


def concat_bboxes(*coords_list: np.ndarray) -> np.ndarray:
    new_coords = np.concatenate(coords_list, axis=0)
    return new_coords