from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Dict, Any
from enum import Enum
import numpy as np
import sympy as sp
import hashlib
import json
from pathlib import Path



@dataclass
class TrainingState:
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    epoch: int
    loss_history: List[float]


@dataclass
class CheckpointConfig:
    data_hash: str
    enabled: bool = False
    cache_dir: str = ".net_cache"
    checkpoint_interval: int = 10
    overwrite: bool = False
