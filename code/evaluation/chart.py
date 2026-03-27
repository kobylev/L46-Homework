import matplotlib.pyplot as plt
import os
import numpy as np

def generate_charts(df, output_dir):
    """
    df: Pandas DataFrame with benchmarking results
    """
    # 1. FPS Comparison
    plt.figure(figsize=(10, 6))
    models = df['Model'].unique()
    fps_means = df.groupby('Model')['FPS'].mean()
    
    plt.bar(fps_means.index, fps_means.values, color=['skyblue', 'orange', 'lightgreen'])
    plt.title('Average FPS across all videos')
    plt.ylabel('FPS')
    plt.xlabel('YOLO Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'fps_comparison.png'))
    plt.close()
    
    # 2. Avg Confidence Comparison
    plt.figure(figsize=(10, 6))
    conf_means = df.groupby('Model')['Avg Confidence'].mean()
    plt.bar(conf_means.index, conf_means.values, color=['salmon', 'teal', 'plum'])
    plt.title('Average Confidence Score across all videos')
    plt.ylabel('Confidence')
    plt.xlabel('YOLO Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'confidence_comparison.png'))
    plt.close()

    # 3. Consistency (Lower is better for object count stability)
    plt.figure(figsize=(10, 6))
    consistency_means = df.groupby('Model')['Consistency'].mean()
    plt.bar(consistency_means.index, consistency_means.values, color=['gold', 'cyan', 'magenta'])
    plt.title('Object Count Consistency (Standard Deviation)')
    plt.ylabel('Std Dev (Lower is more consistent)')
    plt.xlabel('YOLO Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'consistency_comparison.png'))
    plt.close()
