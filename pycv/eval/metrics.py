from typing import Any, Dict, Literal, List, Union, Tuple

import numpy as np


class BaseMetric:
    """
    对一个数据集进行评估
    """
    def process_batch(self, *args, **kwargs) -> None:
        """
        计算一个批次数据的局部中间指标，并储存在buffer中
        """
        raise NotImplementedError
    
    def compute_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        """
        汇总所有数据批次的中间指标，计算全局综合指标
        """
        raise NotImplementedError


class DetMetric(BaseMetric):
    """
    计算mAP, precision, recall
    """
    def __init__(
        self, 
        mode: Literal["bbox", "segm"],
        verbose: bool
    ) -> None:
        self.mode = mode
        self.verbose = verbose
        self.tp_flags = []
        self.fp_flags = []
        self.fn_flags = []
        self.cat_ids = []
        self.img_ids = []
        
        self.metrics = {
            "num_dts_each_cat": [],
            "num_gts_each_cat": [],
            "num_tps_each_cat": [],
            "num_fps_each_cat": [],
            "num_fn_each_cat": [],
            "prec_each_cat": [],
            "rec_each_cat": [],
            "AP_each_cat": [],
            "num_dts_all_cat": 0,
            "num_gts_all_cat": 0,
            "num_tps_all_cat": 0,
            "num_fps_all_cat": 0,
            "num_fn_all_cat": 0,
            "prec_avg_all_cat": 0,
            "rec_avg_all_cat": 0,
            "mAP": 0
        }
    
    def process_batch(
        self, 
        tp_flags: Union[List[bool], np.ndarray], 
        fp_flags: Union[List[bool], np.ndarray], 
        fn_flags: Union[List[bool], np.ndarray], 
        cat_ids: Union[List[int], np.ndarray],
        confs: Union[List[float], np.ndarray],
        img_id: int,
    ) -> None:
        """
        保存单张图片的tp_flags和fn_flags

        Args
        - `tp_flags`: `Union[List[bool], Array[bool]]`, shape `(num_dts_batch, )`
        - `fn_flags`: `Union[List[bool], Array[bool]]`, shape `(num_gts_batch, )`
        - `cat_ids`: `Union[List[int], Array[int]]`, shape `(num_dts_batch, )`
        - `confs`: `Union[List[float], Array[float]]`, shape `(num_dts_batch, )`
        - `img_id`: `int`,
        """
        if isinstance(tp_flags, np.ndarray):
            tp_flags = tp_flags.tolist()
        if isinstance(fp_flags, np.ndarray):
            fp_flags = fp_flags.tolist()
        if isinstance(fn_flags, np.ndarray):
            fn_flags = fn_flags.tolist()
        if isinstance(cat_ids, np.ndarray):
            cat_ids = cat_ids.tolist()
        if isinstance(confs, np.ndarray):
            confs = confs.tolist()
        
        img_ids = [img_id] * len(confs)

        self.img_ids += img_ids
        self.cat_ids += cat_ids
        self.confs += confs
        self.tp_flags += tp_flags
        self.fp_flags += fp_flags
        self.fn_flags += fn_flags
    
    def compute_metrics(self) -> Dict[str, Union[float, List[float]]]:
        pass


def get_AP_prec_rec_single_cat(
    confs: np.ndarray,
    tp_flags: np.ndarray,
    fn_flags: np.ndarray,
) -> Tuple[float, float, float]:
    """
    计算数据集中一个类别的AP, precision, recall

    Args
    - `confs`: `Array[float]`, shape `(num_dts, )`
    - `tp_flags`: `Array[bool]`, shape `(num_dts, )`
    - `fn_flags`: `Array[bool]`, shape `(num_gts, )`
    
    Return
    - `AP`: float
    - `prec`: float
    - `rec`: flaot
    - ``
    """
    # 将dt按照confidence排序, 创建prec-rec curve, 计算AP
    dt_sort_indice = np.argsort(confs)[::-1]
    tp_flags_sort = tp_flags[dt_sort_indice]

    # 用 culmulative sum 计算prec_curve和rec_curve
    # i.e. [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
    num_dts = len(tp_flags)
    num_gts = len(fn_flags)
    ep = 1e-6
    tp_cumsum = np.cumsum(tp_flags_sort.astype(np.uint))
    dt_count_cumsum = np.arange(num_dts)
    prec_curve = tp_cumsum / (dt_count_cumsum + ep)
    rec_curve = tp_cumsum / (num_gts + ep)

    # 用积分计算prec_curve和rec_curve下的面积, 即AP
    # 注意初始点坐标为(prec=1, rec=0)
    prec_curve = np.concatenate([1], prec_curve)
    rec_curve = np.concatenate([0], rec_curve)
    ap = np.trapezoid(prec_curve, rec_curve)
    prec = prec_curve[-1].item()
    rec = rec_curve[-1].item()
    metrics = (ap, prec, rec)


def get_AP_fp_fn(
    gt_insts: np.ndarray,
    gt_img_ids: np.ndarray,
    gt_img_hws: np.ndarray,
    dt_insts: np.ndarray,
    dt_img_ids: np.ndarray,
    dt_img_hws: np.ndarray,
    dt_scores: np.ndarray,
    iou_thres: float
) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray]:
    """
    确定每个dt_insts是否为fp, 每个gt_insts是否为fn, 并计算AP.

    Args
    - `gt_insts`: bbox或者segm
        - bbox: `Array[int]`, shape `(num_gt, 4)`, xywh
        - segm: `Array[uint8]`, shape `(num_gt, max_h, max_w)`, 0/1, 
        pad至最大图片尺寸, 需要用`gt_img_hws`还原
    - `gt_img_ids`: `Array[int]`, shape `(num_gt, )`
    - `gt_img_hws`: `Array[int]`, shape `(num_gt, 2)`, 原图大小, 用于还原mask
    - `dt_insts`: bbox或者segm
        - bbox: `Array[int]`, shape `(num_dt, 4)`, xywh
        - segm: `Array[uint8]`, shape `(num_dt, max_h, max_w)`, 0/1, 
        pad至最大图片尺寸, 需要用`dt_img_hws`还原
    - `dt_img_ids`: `Array[int]`, shape `(num_dt, )`
    - `dt_img_hws`: `Array[int]`, shape `(num_dt, 2)`, 原图大小, 用于还原mask
    - `dt_scores`: `Array[float]`, shape `(num_dt, )`
    - `iou_thres`: `float`

    Returns
    - `metrics`: `Tuple[float, float, float]`, `[AP, prec, rec]`
    - `dt_gt_ids`: `Array[uint]`, 匹配的gt id
    - `dt_tp_flags`: `Array[bool]`, shape `(num_dt, )`
    - `gt_fn_flags`: `Array[bool]`, shape `(num_gt, )`,
    """
    dt_gt_ids = np.empty(len(dt_insts), dtype=np.uint8)
    dt_tp_flags = np.empty(len(dt_insts), dtype=np.uint8)
    gt_fn_flags = np.empty(len(gt_insts), dtype=np.uint8)
    img_ids = set(gt_img_ids) | set(dt_img_ids)

    if len(gt_insts.shape) == 2:
        mode = "bbox"
    elif len(gt_insts) == 3:
        mode = "segm"

    # 遍历图片, 在每张图片上匹配dt和gt, 标记dt是否为tp, gt是否为fn
    for img_id in img_ids:
        gt_indice_imgi = gt_img_ids == img_id
        gt_insts_imgi = gt_insts[gt_indice_imgi]

        dt_indice_imgi = dt_img_ids == img_id
        dt_insts_imgi = dt_insts[dt_indice_imgi]

        img_h, img_w = gt_img_hws[dt_indice_imgi][0].tolist()

        if mode == "bbox":
            iou = get_iou_bbox(gt_insts_imgi, dt_insts_imgi)
        else:
            iou = get_iou_segm(gt_insts_imgi, dt_insts_imgi)
        
        dt_gt_ids_imgi, dt_tp_flags_imgi, gt_fn_flags_imgi = assign_gt_to_dt(
            iou, iou_thres, True
        )

        dt_gt_ids[dt_indice_imgi] = dt_gt_ids_imgi
        dt_tp_flags[dt_indice_imgi] = dt_tp_flags_imgi
        gt_fn_flags[gt_indice_imgi] = gt_fn_flags_imgi
    
    # 将dt按照confidence排序, 创建prec-rec curve, 计算AP
    dt_sort_indice = np.argsort(dt_scores)[::-1]
    dt_tp_flags_sort = dt_tp_flags[dt_sort_indice]

    # 用 culmulative sum 计算prec_curve和rec_curve
    # i.e. [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
    num_dts = len(dt_insts)
    num_gts = len(gt_insts)
    ep = 1e-6
    tp_cumsum = np.cumsum(dt_tp_flags_sort.astype(np.uint))
    prec_curve = tp_cumsum / (num_dts + ep)
    rec_curve = tp_cumsum / (num_gts + ep)

    # 用积分计算prec_curve和rec_curve下的面积, 即AP
    # 注意初始点坐标为(prec=1, rec=0)
    prec_curve = np.concatenate([1], prec_curve)
    rec_curve = np.concatenate([0], rec_curve)
    ap = np.trapz(prec_curve, rec_curve)
    prec = prec_curve[-1].item()
    rec = rec_curve[-1].item()
    metrics = (ap, prec, rec)

    return metrics, dt_gt_ids, dt_tp_flags, gt_fn_flags