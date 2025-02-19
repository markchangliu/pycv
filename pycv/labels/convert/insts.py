import copy
import json
import os
from typing import Union, Dict, Tuple

import numpy as np

from pycv.structures import DetInsts, SegmInsts, bitmask_to_polygon


def insts2labelme(
    segm_insts: Union[SegmInsts, DetInsts],
    img_name: str,
    export_json_p: Union[os.PathLike, str],
    img_hw: Tuple[int, int] = (0, 0),
    cat_id_name_dict: Dict[int, str] = {0: "object"}
) -> None:
    labelme_template = {
        "version": "4.6.0",
        "flags": {},
        "shapes": [],
        "imagePath": "",
        "imageHeight": -1,
        "imageWidth": -1,
        "imageData": None,
        "fillColor": [255, 0, 0, 128],
        "lineColor": [0, 255, 0, 128]
    }

    shape_template = {
        "line_color": None,
        "fill_color": None,
        "label": "",
        "points": [],
        "group_id": -1,
        "shape_type": "",
        "flags": {}
    }

    labelme_dict = copy.deepcopy(labelme_template)

    for i, inst in enumerate(segm_insts):
        label_id = inst.cats.item()

        if isinstance(inst, SegmInsts):
            bitmask = inst.masks[0]
            polygon = bitmask_to_polygon(bitmask)
            shape_type = "polygon"
        elif isinstance(inst, DetInsts):
            bbox = inst.bboxes[0].tolist()
            x1, y1, x2, y2 = bbox
            polygon = [[x1, y1], [x2, y2]]
            shape_type = "rectangle"

        shape = copy.deepcopy(shape_template)
        shape["shape_type"] = shape_type
        shape["points"] = polygon
        shape["label"] = cat_id_name_dict[label_id]
        shape["group_id"] = None
        labelme_dict["shapes"].append(shape)

    labelme_dict["imagePath"] = img_name
    labelme_dict["imageData"] = None
    
    if isinstance(segm_insts, SegmInsts):
        labelme_dict["imageHeight"] = segm_insts.masks.shape[1]
        labelme_dict["imageWidth"] = segm_insts.masks.shape[2]
    else:
        labelme_dict["imageHeight"] = int(img_hw[0])
        labelme_dict["imageWidth"] = int(img_hw[1])
    
    with open(export_json_p, "w") as f:
        json.dump(labelme_dict, f)

def insts2npz(
    insts: Union[SegmInsts, DetInsts],
    export_npz_p: Union[os.PathLike, str]
) -> None:
    if isinstance(insts, DetInsts):
        bboxes = insts.bboxes
        scores = insts.scores
        np.savez_compressed(export_npz_p, bboxes=bboxes, scores=scores)
    elif isinstance(insts, SegmInsts):
        bboxes = insts.bboxes
        scores = insts.scores
        masks = insts.masks.astype(np.int32)
        np.savez_compressed(
            export_npz_p, bboxes = bboxes, scores = scores, masks = masks
        )