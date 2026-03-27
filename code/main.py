import os
import pandas as pd
from config import MODELS, VIDEOS, ASSETS_DIR, RESULTS_DIR, VIDEO_RESULTS_DIR, DEVICE
from detectors.yolo_detector import YOLODetector
from video_processor import process_video
from evaluation.metrics import calculate_video_metrics
from evaluation.table import export_results
from evaluation.chart import generate_charts
from evaluation.conclusion import generate_conclusion

def main():
    # Ensure results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if not os.path.exists(VIDEO_RESULTS_DIR):
        os.makedirs(VIDEO_RESULTS_DIR)
        
    all_video_results = []
    
    for model_name in MODELS:
        print(f"\n--- Initializing Model: {model_name} on {DEVICE} ---")
        try:
            detector = YOLODetector(model_name, device=DEVICE)
            params_m = detector.get_params_m()
            
            for video_name in VIDEOS:
                video_path = os.path.join(ASSETS_DIR, video_name)
                if not os.path.exists(video_path):
                    print(f"Skipping {video_name}, file not found at {video_path}")
                    continue
                
                # Output path for annotated video in video_result folder
                output_video_path = os.path.join(VIDEO_RESULTS_DIR, f"{model_name.replace('.pt','')}_{video_name}")
                
                # 1. Process Video (Inference + Annotation)
                frame_results = process_video(detector, video_path, output_video_path)
                
                if frame_results:
                    # 2. Calculate Video-level Metrics
                    video_metrics = calculate_video_metrics(frame_results, model_name, video_name, params_m)
                    all_video_results.append(video_metrics)
                    
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            
    if not all_video_results:
        print("No results to export.")
        return
        
    # 3. Export Summary Table (CSV + Markdown)
    results_df = export_results(all_video_results, RESULTS_DIR)
    
    # 4. Generate Charts
    generate_charts(results_df, RESULTS_DIR)
    
    # 5. Generate Automated Conclusion
    generate_conclusion(results_df, RESULTS_DIR)
    
    print("\n--- All Benchmarks Completed Successfully ---")
    print(f"Check results in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
