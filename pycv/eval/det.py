import copy
import numpy as np


def assign_gt_to_pred(
    score_mat: np.ndarray,
    thres: float
) -> np.ndarray:
    """
    为 pred 匹配 gt，参考：
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/matcher.py
    

    Args
    - `score_mat`: `Array[float]`, shape `(num_pred, num_gt)`
    - `thres`: `float`

    Returns
    - `assigned_gt_ids`: `Array[long]`, shape `(num_pred, )`,
    最大分数的 gt 索引
    - `assigned_labels`: `Array[bool]`, shape `(num_pred, )`, 
    最大分数是否超过 `thres`
    """
    num_pred, num_gt = score_mat.shape
    score_mat = copy.deepcopy(score_mat)

    assigned_gt_ids = np.ones((num_pred, ), dtype=np.long) * -1
    assigned_labels = np.zeros((num_pred, ), dtype=np.bool_)

    if num_pred == 0 or np.max(score_mat) < thres or num_gt == 0:
        return assigned_gt_ids, assigned_labels

    # 为 pred 匹配分数最大的 gt
    max_scores_gt_ids = np.argmax(score_mat, axis=1)
    max_scores = score_mat[range(num_pred), max_scores_gt_ids]

    # 讲分数大于 thres 的 pred label 设为 True
    assigned_gt_ids = max_scores_gt_ids
    assigned_labels[max_scores>thres] = True
    
    return assigned_gt_ids, assigned_labels