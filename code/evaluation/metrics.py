import numpy as np
import pandas as pd

def calculate_video_metrics(frame_results, model_name, video_name, params_m):
    """
    frame_results: list of dicts with count, inference_time_ms, avg_confidence, high_conf_ratio
    """
    counts = [fr['count'] for fr in frame_results]
    inf_times = [fr['inference_time_ms'] for fr in frame_results]
    avg_confs = [fr['avg_confidence'] for fr in frame_results]
    high_conf_ratios = [fr['high_conf_ratio'] for fr in frame_results]
    
    avg_inf_time = np.mean(inf_times) if inf_times else 0.0
    fps = 1000 / avg_inf_time if avg_inf_time > 0 else 0.0
    total_detections = sum(counts)
    avg_confidence = np.mean(avg_confs) if avg_confs else 0.0
    avg_high_conf_ratio = np.mean(high_conf_ratios) if high_conf_ratios else 0.0
    consistency = np.std(counts) if counts else 0.0
    
    return {
        "Model": model_name,
        "Video": video_name,
        "Avg Inference (ms)": avg_inf_time,
        "FPS": fps,
        "Total Detections": total_detections,
        "Avg Confidence": avg_confidence,
        "High Confidence Ratio": avg_high_conf_ratio,
        "Consistency": consistency,
        "Params (M)": params_m
    }
