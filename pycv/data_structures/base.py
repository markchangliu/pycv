from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict


class DataType(Enum):
    BBOXES: str = "bounding_bboxes"
    MASKS: str = "segmentation_masks"


@dataclass
class BaseStructure:

    def validate(self) -> None:
        raise NotImplementedError
    
