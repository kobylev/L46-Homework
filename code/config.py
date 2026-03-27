import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'Assets')
CODE_DIR = os.path.join(BASE_DIR, 'code')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
VIDEO_RESULTS_DIR = os.path.join(BASE_DIR, 'video_result')

# Models to benchmark
MODELS = ["yolov8n.pt", "yolov9c.pt", "yolo11n.pt"]

# Videos to benchmark
VIDEOS = ["Ambulance.mp4", "rome.mp4"]

# Device (mps, cuda, cpu)
DEVICE = "cuda"  # CUDA is available
