import cv2
import os

def process_video(detector, video_path, output_path):
    """
    Processes a video frame by frame, runs inference, and saves annotated results.
    Returns a list of frame-level results.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return []
    
    # Video properties for saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, orig_fps, (width, height))
    
    frame_results = []
    frame_idx = 0
    
    print(f"Processing: {os.path.basename(video_path)} with {detector.model_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference
        result = detector.predict(frame)
        
        # Save frame metrics
        frame_results.append({
            "count": result['count'],
            "inference_time_ms": result['inference_time_ms'],
            "avg_confidence": result['avg_confidence'],
            "high_conf_ratio": result['high_conf_ratio']
        })
        
        # Write annotated frame
        out.write(result['annotated_frame'])
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames...")
            
    cap.release()
    out.release()
    return frame_results
