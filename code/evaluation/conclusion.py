import os

def generate_conclusion(df, output_dir):
    """
    df: Pandas DataFrame with benchmarking results
    """
    # Average across all videos for each model
    summary = df.groupby('Model').mean(numeric_only=True)
    
    # Identify winners for different categories
    # Speed winner: Highest FPS
    speed_winner = summary['FPS'].idxmax()
    # Quality winner: Highest Avg Confidence
    quality_winner = summary['Avg Confidence'].idxmax()
    # Stability winner: Lowest Consistency (Standard Deviation)
    stability_winner = summary['Consistency'].idxmin()
    # Efficiency winner: Lowest Params (M)
    efficiency_winner = summary['Params (M)'].idxmin()
    
    conclusion_path = os.path.join(output_dir, 'conclusion.txt')
    with open(conclusion_path, 'w') as f:
        f.write("--- YOLO Benchmarking Evaluation --- \n\n")
        f.write(f"Speed Winner: {speed_winner} (Highest FPS)\n")
        f.write(f"Quality Winner: {quality_winner} (Highest Average Confidence)\n")
        f.write(f"Stability Winner: {stability_winner} (Lowest Variation in Object Count)\n")
        f.write(f"Efficiency Winner: {efficiency_winner} (Fewest Parameters)\n\n")
        
        f.write("Evaluation Summary:\n")
        f.write("--------------------\n")
        for model in summary.index:
            f.write(f"- {model}: avg {summary.loc[model, 'FPS']:.2f} FPS with {summary.loc[model, 'Avg Confidence']:.2f} confidence score.\n")
    
    print(f"Conclusion generated at {conclusion_path}")
