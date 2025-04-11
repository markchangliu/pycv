import os
from dataclasses import dataclass


@dataclass
class Img:
    img_p: str
    img_h: int
    img_w: int