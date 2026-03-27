import time
from ultralytics import YOLO
import torch
import os

def get_model_parameters(model):
    """
    Returns total parameters of the model (in millions).
    """
    return sum(p.numel() for p in model.parameters()) / 1e6

def benchmark_model(model_name, video_path, device='cpu'):
    """
    Benchmarks a single YOLO model on a video.
    Returns average inference time (ms), total objects, and model size (M params).
    """
    print(f"--- Benchmarking {model_name} ---")
    
    # Load model
    model = YOLO(model_name).to(device)
    model_params = get_model_parameters(model.model)
    
    # Run inference
    # verbose=False reduces console noise
    results = model.predict(source=video_path, device=device, verbose=False)
    
    total_time_ms = 0
    total_objects = 0
    frame_count = len(results)
    
    for res in results:
        # Summing the inference time (preprocess + inference + postprocess)
        # Ultralytics results.speed is a dict in ms
        total_time_ms += res.speed['inference']
        total_objects += len(res.boxes)
        
    avg_inference_time = total_time_ms / frame_count if frame_count > 0 else 0
    
    return {
        "model": model_name,
        "avg_time_ms": avg_inference_time,
        "fps": 1000 / avg_inference_time if avg_inference_time > 0 else 0,
        "total_objects": total_objects,
        "params_m": model_params
    }

def run_benchmarks(models, video_path, device='cpu'):
    """
    Runs benchmarks for a list of models.
    """
    benchmark_results = []
    for model_name in models:
        try:
            res = benchmark_model(model_name, video_path, device)
            benchmark_results.append(res)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            
    return benchmark_results
