from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class FrameResult:
    frame_index: int
    detections_count: int
    inference_time: float
    avg_confidence: float
    high_conf_ratio: float  # ratio of detections > 0.7

@dataclass
class VideoResult:
    model_name: str
    video_name: str
    avg_inference_time: float
    fps: float
    total_detections: int
    avg_confidence: float
    high_conf_ratio: float
    consistency: float  # std deviation of object counts
    params_m: float

class BaseDetector(ABC):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def predict(self, frame):
        pass

    @abstractmethod
    def get_params_m(self) -> float:
        pass
