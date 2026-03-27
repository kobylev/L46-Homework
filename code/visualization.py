import matplotlib.pyplot as plt
import os

def plot_speed_comparison(results, output_path):
    """
    Creates a bar chart comparing FPS across different YOLO models.
    """
    models = [res['model'] for res in results]
    fps_values = [res['fps'] for res in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, fps_values, color=['skyblue', 'orange', 'lightgreen'])
    
    # Adding data labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
    
    plt.xlabel('YOLO Model')
    plt.ylabel('Inference Speed (FPS)')
    plt.title('YOLOv8 vs YOLOv10 Inference Speed Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()
