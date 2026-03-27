import cv2
import os

def extract_frame(video_path, output_image_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return False
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_image_path, frame)
        print(f"Extracted frame {frame_number} to {output_image_path}")
    else:
        print(f"Could not read frame {frame_number} from {video_path}")
    
    cap.release()
    return ret

def main():
    video_dir = "video_result"
    output_dir = "frame_comparison"
    frame_no = 100
    
    for filename in os.listdir(video_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_dir, filename)
            output_name = filename.replace(".mp4", f"_frame{frame_no}.jpg")
            output_path = os.path.join(output_dir, output_name)
            extract_frame(video_path, output_path, frame_no)

if __name__ == "__main__":
    main()
