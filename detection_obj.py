from dataclasses import dataclass
from typing import List, Any


@dataclass
class DetectionObject:
    bbox: List[Any]
    id_: int
    uniq_id: str = str()
    hit_count: int = 0
