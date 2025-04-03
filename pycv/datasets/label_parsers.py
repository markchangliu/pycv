import json
import os
from typing import Union, Dict, Any

import numpy as np
import pycocotools.mask as pycocomask

from pycv.data_structures.det_data import DetData
from pycv.data_structures.bboxes import BBoxes, BBoxFormat
from pycv.data_structures.masks import Masks, MaskFormat
from pycv.data_structures.insts import Insts, InstsType


def parse_labelme_json(
    labelme_p: Union[str, os.PathLike],
    cat_name_id_dict: Dict[str, int],
    img_prefix: Union[str, os.PathLike]
) -> DetData:
    with open(labelme_p, "r") as f:
        labelme_dict = json.load(f)
    
    img_name = labelme_dict["imagePath"]
    img_p = os.path.join(img_prefix, img_name)
    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]

    img_tags = []
    for k, v in labelme_dict["flags"]:
        if v:
            img_tags.append(k)

    bboxes = []
    masks = []
    cat_ids = []
    all_shape_tags = []

    grouped_shapes: Dict[Any, list] = {} # {shape_group: [shapes]}

    for shape in labelme_dict["shapes"]:
        if shape["group_id"] in grouped_shapes.keys():
            grouped_shapes[shape["group_id"]].append(shape)
        else:
            grouped_shapes[shape["group_id"]] = [shape]
    
    # 把group_id为None的shape当作个体
    for shape in grouped_shapes[None]:
        if shape["shape_type"] == "rectangle":
            x1y1, x2y2 = shape["points"]
            x1, y1 = x1y1
            x2, y2 = x2y2
            bbox = [x1, y1, x2, y2]
            mask = None
        elif shape["shape_type"] == "polygon":
            poly = np.asarray(shape["points"], dtype=np.int32) # (num_points, 2)
            poly = poly.flatten().tolist() # (num_points * 2, )
            rle = pycocomask.frPyObjects([poly], img_h, img_w)
            mask = pycocomask.decode(rle) # (img_h, img_w, 1)
            mask = np.transpose(mask, (2, 0, 1)) # (1, img_h, img_w)
            x1, y1, w, h = pycocomask.toBbox(rle).flatten().tolist()
            x2, y2 = x1 + w, y1 + h
            bbox = [x1, y1, x2, y2]
        
        cat_id = cat_name_id_dict[shape["label"]]
        shape_tags = shape["description"].split(",")
        shape_tags = [t.strip() for t in shape_tags]

        cat_ids.append(cat_id)
        bboxes.append(bbox)
        masks.append(mask)
        all_shape_tags.append(shape_tags)
    
    del grouped_shapes[None]

    # 把group_id为相同数字的shape融合起来
    # 仅支持polygon
    # 融合后的mask为所有单独mask的union
    # 融合后的bbox为融合后mask的bbox
    # 融合后的cat_id为首个shape的cat_id
    # 融合后的tags为所有单独shape的tags集合
    for g_id, shapes in grouped_shapes.items():
        rles_obj = []
        tags_obj = []

        for shape in shapes:
            if shape["shape_type"] == "rectangle":
                raise NotImplementedError
            
            poly = np.asarray(shape["points"], dtype=np.int32) # (num_points, 2)
            poly = poly.flatten().tolist() # (num_points * 2, )
            rle = pycocomask.frPyObjects(poly, img_h, img_w)
            rles_obj.append(rle)

            shape_tags = shape["description"].split(",")
            shape_tags = [t.strip() for t in shape_tags]

            tags_obj += shape_tags
        
        rle_obj = pycocomask.merge(rles_obj)
        mask = pycocomask.decode(rle_obj) # (img_h, img_w, 1)
        mask = np.transpose(mask, (2, 0, 1)) # (1, img_h, img_w)
        x1, y1, w, h = pycocomask.toBbox(rle).flatten().tolist()
        x2, y2 = x1 + w, y1 + h
        bbox = [x1, y1, x2, y2]

        cat_id = cat_name_id_dict[shapes[0]["label"]]
        tags_obj = list(set(tags_obj))

        bboxes.append(bbox)
        masks.append(mask)
        cat_ids.append(cat_id)
        all_shape_tags.append(tags_obj)

    bboxes = BBoxes(np.asarray(bboxes), BBoxFormat.XYXY)
    masks = Masks(masks, (img_h, img_w), MaskFormat.BINARY)
    cat_ids = np.asarray(cat_ids)
    confs = np.ones(len(cat_ids))
    insts = Insts(confs, cat_ids, bboxes, masks)
    det_data = DetData(img_p, insts, img_tags, all_shape_tags)

    return det_data