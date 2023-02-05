from dataclasses import dataclass
import numpy as np


@dataclass
class Source:
    stream: str
    stream_id: int
    image: np.ndarray = 0
    image_to_detect: np.ndarray = 0
