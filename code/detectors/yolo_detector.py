from ultralytics import YOLO
from .base_detector import BaseDetector
import time
import numpy as np

class YOLODetector(BaseDetector):
    def _load_model(self):
        self.model = YOLO(self.model_path).to(self.device)

    def predict(self, frame):
        # We perform inference and return frame-level metrics
        start_time = time.time()
        results = self.model.predict(frame, device=self.device, verbose=False)[0]
        end_time = time.time()
        
        # Inference time from model.predict results
        # Note: results.speed is in ms
        inf_time_ms = results.speed['inference']
        
        boxes = results.boxes
        confs = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
        
        count = len(boxes)
        avg_conf = np.mean(confs) if count > 0 else 0.0
        high_conf_count = np.sum(confs >= 0.7) if count > 0 else 0
        high_conf_ratio = high_conf_count / count if count > 0 else 0.0
        
        return {
            "count": count,
            "inference_time_ms": inf_time_ms,
            "avg_confidence": float(avg_conf),
            "high_conf_ratio": float(high_conf_ratio),
            "annotated_frame": results.plot()
        }

    def get_params_m(self) -> float:
        return sum(p.numel() for p in self.model.model.parameters()) / 1e6
